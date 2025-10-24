import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency in some envs
    wandb = None

from src.data import create_tiny_shakespeare_dataloader
from src.diff_utils import mask_tokens_from_timesteps, sample_timesteps
from src.model import DiffusionModelConfig, TextDiffusionModel

TS_PATH = Path("data/tiny_shakespeare.txt")


def main():
    parser = argparse.ArgumentParser()
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
        help="Tokenizer to use for dataset preparation",
    )
    parser.add_argument(
        "--tiktoken-encoding",
        type=str,
        default="gpt2",
        help="tiktoken encoding name when using the tiktoken tokenizer",
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
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloader = create_tiny_shakespeare_dataloader(
        text_path=TS_PATH,
        seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        shuffle=True,
        tokenizer=args.tokenizer,
        tiktoken_encoding=args.tiktoken_encoding,
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
            },
        )

    step = 0
    for epoch in range(args.num_epochs):
        for input_ids in dataloader:
            step += 1

            input_ids = input_ids.to(device)
            batch_size = input_ids.size(0)

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
                    originals = []
                    masked_prompts = []
                    generations = []

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

                        max_gen_timestep = int(gen_timesteps.max().item())
                        for t in reversed(range(max_gen_timestep + 1)):
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
                                probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                            sampled = torch.multinomial(
                                probs.view(-1, probs.size(-1)), 1
                            ).view_as(current_tokens)
                            current_tokens = torch.where(generation_mask, sampled, current_tokens)

                        for i in range(total_samples):
                            originals.append(dataset.decode(reference[i].cpu()))
                            masked_prompts.append(dataset.decode(initial_masked[i].cpu()))
                            generations.append(dataset.decode(current_tokens[i].cpu()))
                model.train()

                if originals:
                    if wandb_run is not None and wandb is not None:
                        table = wandb.Table(
                            columns=["original", "masked", "generated"],
                            data=list(zip(originals, masked_prompts, generations)),
                        )
                        wandb_run.log({"samples": table}, step=step)
                    else:
                        print("\n=== Generations ===")
                        for idx, (orig, masked, gen) in enumerate(
                            zip(originals, masked_prompts, generations), 1
                        ):
                            print(f"[{idx}]")
                            print(f"  original : {orig}")
                            print(f"  masked   : {masked}")
                            print(f"  generated: {gen}")
                        print("===================\n")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
