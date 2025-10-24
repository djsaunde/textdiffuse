import argparse
import math
import warnings
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency in some envs
    wandb = None

try:  # pragma: no cover - pydantic optional dependency
    from pydantic import warnings as pydantic_warnings

    warnings.filterwarnings(
        "ignore",
        category=pydantic_warnings.UnsupportedFieldAttributeWarning,
    )
except Exception:
    pass

from src.data import create_fineweb_dataloader, create_tiny_shakespeare_dataloader
from src.diff_utils import (
    confidence_based_remask,
    mask_tokens_from_timesteps,
    sample_timesteps,
    uniform_random_remask,
)
from src.model import DiffusionModelConfig, TextDiffusionModel

TS_PATH = Path("data/tiny_shakespeare.txt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=("fineweb", "tiny"), default="fineweb")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--num-timesteps", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="textdiffuse")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument(
        "--wandb-mode",
        type=str,
        choices=("online", "offline"),
        default="online",
        help="W&B mode when logging is enabled",
    )
    parser.add_argument("--generate-every", type=int, default=0)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument(
        "--tokenizer",
        type=str,
        choices=("tiktoken", "char"),
        default="tiktoken",
        help="Tokenizer to use (only relevant for the Tiny Shakespeare dataset)",
    )
    parser.add_argument(
        "--tiktoken-encoding",
        type=str,
        default="gpt2",
        help="tiktoken encoding name",
    )
    parser.add_argument(
        "--generation-min-frac",
        type=float,
        default=0.25,
        help="Lower bound (fraction of schedule) for generation timesteps",
    )
    parser.add_argument(
        "--generation-max-frac",
        type=float,
        default=0.75,
        help="Upper bound (fraction of schedule) for generation timesteps",
    )
    parser.add_argument(
        "--generation-remask",
        type=str,
        choices=("none", "confidence", "uniform"),
        default="confidence",
        help="Remasking strategy to use during generation refinement",
    )
    parser.add_argument(
        "--generation-confidence-threshold",
        type=float,
        default=0.85,
        help="Confidence threshold for confidence-based remasking",
    )
    parser.add_argument(
        "--generation-min-mask",
        type=int,
        default=0,
        help="Minimum number of tokens to keep masked when using confidence remasking",
    )
    parser.add_argument(
        "--generation-remask-ratio",
        type=float,
        default=0.2,
        help="Fraction of tokens to re-mask when using uniform remasking",
    )
    parser.add_argument(
        "--fineweb-dir",
        type=str,
        default="data/fineweb",
        help="Directory containing FineWeb parquet shards",
    )
    parser.add_argument(
        "--fineweb-num-shards",
        type=int,
        default=1,
        help="Number of sequential FineWeb shards to load",
    )
    parser.add_argument(
        "--fineweb-shard-offset",
        type=int,
        default=0,
        help="Starting shard index for FineWeb shards",
    )
    parser.add_argument(
        "--fineweb-docs-per-shard",
        type=int,
        default=None,
        help="Optional cap on number of documents to read per FineWeb shard",
    )
    parser.add_argument(
        "--fineweb-max-tokens",
        type=int,
        default=None,
        help="Optional cap on total tokens to load from FineWeb",
    )
    parser.add_argument(
        "--fineweb-no-download",
        action="store_true",
        help="Disable automatic downloading of missing FineWeb shards",
    )
    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == "tiny":
        dataloader = create_tiny_shakespeare_dataloader(
            text_path=TS_PATH,
            seq_len=args.max_seq_len,
            batch_size=args.batch_size,
            shuffle=True,
            tokenizer=args.tokenizer,
            tiktoken_encoding=args.tiktoken_encoding,
        )
    else:
        dataloader = create_fineweb_dataloader(
            data_dir=args.fineweb_dir,
            seq_len=args.max_seq_len,
            batch_size=args.batch_size,
            encoding_name=args.tiktoken_encoding,
            num_shards=args.fineweb_num_shards,
            shard_offset=args.fineweb_shard_offset,
            docs_per_shard=args.fineweb_docs_per_shard,
            max_tokens=args.fineweb_max_tokens,
            auto_download=not args.fineweb_no_download,
            shuffle=True,
        )
    dataset = dataloader.dataset

    config = DiffusionModelConfig(
        vocab_size=dataset.vocab_size,
        max_seq_len=args.max_seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
    )
    print(config)

    model = TextDiffusionModel(config).to(device)
    model.train()
    optim = torch.optim.AdamW(params=model.parameters(), lr=args.lr)

    mask_token_id = dataset.mask_token_id
    pad_token_id = dataset.pad_token_id

    wandb_run = None
    if args.wandb:
        if wandb is None:
            msg = "wandb is not installed. Disable --wandb or install the dependency."
            raise RuntimeError(msg)
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            mode=args.wandb_mode,
            config={
                "dataset": args.dataset,
                "tokenizer": args.tokenizer,
                "tiktoken_encoding": args.tiktoken_encoding,
                "max_seq_len": args.max_seq_len,
                "d_model": args.d_model,
                "n_heads": args.n_heads,
                "n_layers": args.n_layers,
                "batch_size": args.batch_size,
                "num_epochs": args.num_epochs,
                "num_timesteps": args.num_timesteps,
                "lr": args.lr,
                "generation_remask": args.generation_remask,
                "generation_confidence_threshold": args.generation_confidence_threshold,
                "generation_min_mask": args.generation_min_mask,
                "generation_remask_ratio": args.generation_remask_ratio,
                "fineweb_num_shards": args.fineweb_num_shards,
                "fineweb_shard_offset": args.fineweb_shard_offset,
                "fineweb_docs_per_shard": args.fineweb_docs_per_shard,
                "fineweb_max_tokens": args.fineweb_max_tokens,
            },
        )

    step = 0
    tokens_seen = 0
    for epoch in range(args.num_epochs):
        for input_ids in dataloader:
            step += 1

            input_ids = input_ids.to(device)
            batch_size = input_ids.size(0)
            tokens_seen += int(batch_size * input_ids.size(1))

            timesteps = sample_timesteps(
                num_timesteps=args.num_timesteps,
                batch_size=batch_size,
                device=device,
            )

            masked_tokens, mask, _ = mask_tokens_from_timesteps(
                input_ids=input_ids,
                timesteps=timesteps,
                num_timesteps=args.num_timesteps,
                mask_token_id=mask_token_id,
            )

            num_masked = mask.sum().item()
            if num_masked == 0:
                continue

            attention_mask = input_ids != pad_token_id
            logits = model(
                masked_tokens,
                timesteps=timesteps,
                attention_mask=attention_mask,
            )

            target_tokens = input_ids[mask]
            pred_logits = logits[mask]

            loss = F.cross_entropy(pred_logits, target_tokens)

            optim.zero_grad()
            loss.backward()
            optim.step()

            mask_fraction = num_masked / (batch_size * input_ids.size(1))

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/loss": loss.item(),
                        "train/mask_fraction": mask_fraction,
                        "train/timestep_mean": timesteps.float().mean().item(),
                        "train/timestep_std": timesteps.float().std(unbiased=False).item(),
                        "train/lr": optim.param_groups[0]["lr"],
                        "train/tokens": tokens_seen,
                    },
                    step=step,
                )

            if step % args.log_every == 0:
                print(
                    f"epoch={epoch} step={step} loss={loss.item():.4f} "
                    f"avg_mask={mask_fraction:.3f}"
                )

            if args.generate_every > 0 and step % args.generate_every == 0:
                model.eval()
                with torch.no_grad():
                    total_samples = min(args.num_generations, len(dataset))
                    originals: List[str] = []
                    masked_prompts: List[str] = []
                    generations: List[str] = []
                    initial_mask_fracs: List[float] = []
                    final_mask_fracs: List[float] = []
                    timesteps_logged: List[int] = []
                    confidences_logged: List[float] = []

                    if total_samples > 0:
                        sample_indices = torch.randint(
                            0,
                            len(dataset),
                            (total_samples,),
                        )
                        reference = torch.stack(
                            [dataset[int(idx)] for idx in sample_indices]
                        ).to(device)

                        min_frac = max(0.0, min(args.generation_min_frac, 1.0))
                        max_frac = max(0.0, min(args.generation_max_frac, 1.0))
                        if max_frac < min_frac:
                            min_frac, max_frac = max_frac, min_frac

                        low = int(min_frac * args.num_timesteps)
                        high = int(max_frac * args.num_timesteps)
                        high = max(low + 1, high)

                        gen_timesteps = torch.randint(
                            low,
                            high,
                            (total_samples,),
                            device=device,
                            dtype=torch.long,
                        )

                        masked_tokens, generation_mask, _ = mask_tokens_from_timesteps(
                            input_ids=reference,
                            timesteps=gen_timesteps,
                            num_timesteps=args.num_timesteps,
                            mask_token_id=mask_token_id,
                        )

                        initial_masked = masked_tokens.clone()
                        attention = torch.ones_like(reference, dtype=torch.bool)
                        current_tokens = masked_tokens
                        current_mask = generation_mask.clone()

                        initial_mask_fracs = (
                            generation_mask.float().mean(dim=1).cpu().tolist()
                        )

                        max_gen_timestep = int(gen_timesteps.max().item())
                        last_confidences = None
                        for t in reversed(range(max_gen_timestep + 1)):
                            active = gen_timesteps >= t
                            if not torch.any(active):
                                continue

                            step_mask = current_mask & active.unsqueeze(1)
                            if not torch.any(step_mask):
                                continue

                            timestep_batch = torch.full(
                                (total_samples,), t, device=device, dtype=torch.long
                            )
                            logits_step = model(
                                current_tokens,
                                timesteps=timestep_batch,
                                attention_mask=attention,
                            )
                            probs = torch.softmax(logits_step, dim=-1)
                            if pad_token_id is not None:
                                probs[..., pad_token_id] = 0.0
                            if mask_token_id is not None:
                                probs[..., mask_token_id] = 0.0
                            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                            sampled = torch.multinomial(
                                probs.view(-1, probs.size(-1)), 1
                            ).view_as(current_tokens)
                            current_tokens = torch.where(step_mask, sampled, current_tokens)

                            if args.generation_remask == "confidence":
                                current_tokens, current_mask, last_confidences = (
                                    confidence_based_remask(
                                        logits=logits_step,
                                        tokens=current_tokens,
                                        current_mask=current_mask,
                                        mask_token_id=mask_token_id,
                                        confidence_threshold=args.generation_confidence_threshold,
                                        min_mask_tokens=args.generation_min_mask,
                                    )
                                )
                                current_mask &= generation_mask
                            elif args.generation_remask == "uniform":
                                current_tokens, current_mask = uniform_random_remask(
                                    tokens=current_tokens,
                                    current_mask=current_mask,
                                    remask_ratio=args.generation_remask_ratio,
                                    mask_token_id=mask_token_id,
                                    maskable=generation_mask,
                                )
                                current_mask &= generation_mask
                            else:
                                current_mask &= generation_mask

                        final_mask_fracs = (
                            current_mask.float().mean(dim=1).cpu().tolist()
                        )
                        if last_confidences is not None:
                            confidences_logged = (
                                last_confidences.detach().cpu().mean(dim=1).tolist()
                            )
                        else:
                            confidences_logged = [float("nan")] * total_samples
                        timesteps_logged = gen_timesteps.detach().cpu().tolist()

                        for i in range(total_samples):
                            originals.append(dataset.decode(reference[i].cpu()))
                            masked_prompts.append(dataset.decode(initial_masked[i].cpu()))
                            generations.append(dataset.decode(current_tokens[i].cpu()))
                model.train()

                if originals:
                    if wandb_run is not None and wandb is not None:
                        table = wandb.Table(
                            columns=[
                                "timestep",
                                "mask_frac_initial",
                                "mask_frac_final",
                                "confidence",
                                "original",
                                "masked",
                                "generated",
                            ],
                            data=[
                                [
                                    timesteps_logged[i],
                                    initial_mask_fracs[i],
                                    final_mask_fracs[i],
                                    confidences_logged[i],
                                    originals[i],
                                    masked_prompts[i],
                                    generations[i],
                                ]
                                for i in range(len(generations))
                            ],
                        )
                        wandb_run.log({"samples": table}, step=step)
                    else:
                        print("\n=== Generation Samples ===")
                        for idx, (orig, masked, gen, tstep, init_frac, final_frac, conf) in enumerate(
                            zip(
                                originals,
                                masked_prompts,
                                generations,
                                timesteps_logged,
                                initial_mask_fracs,
                                final_mask_fracs,
                                confidences_logged,
                            ),
                            1,
                        ):
                            print(
                                f"[Sample {idx}] timestep={int(tstep)} initial_mask={init_frac:.3f} final_mask={final_frac:.3f}"
                            )
                            if not math.isnan(conf):
                                print(f"  last_confidence={conf:.3f}")
                            print("  -- Original --")
                            print(f"  {orig}")
                            print("  -- Masked Input --")
                            print(f"  {masked}")
                            print("  -- Generated Output --")
                            print(f"  {gen}")
                            print("  ---------------------")
                        print("===========================\n")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
