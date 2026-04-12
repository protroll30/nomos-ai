"""LoRA SFT with Unsloth on data/train.jsonl (OpenAI-style messages).

Default base weights: Llama 3.1 8B Instruct (4-bit). Requires CUDA GPU.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description="Unsloth LoRA fine-tuning for Nomos JSONL.")
    ap.add_argument("--train", type=Path, default=ROOT / "data" / "train.jsonl")
    ap.add_argument("--eval", type=Path, default=ROOT / "data" / "eval.jsonl")
    ap.add_argument(
        "--model-name",
        default="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        help="HF id. Default: Llama 3.1 8B Instruct (chat), 4-bit (Unsloth). Use base only if you intend pretrain-style LM.",
    )
    ap.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "nomos-lora")
    ap.add_argument("--max-seq-length", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=16)
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from unsloth import FastLanguageModel
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        print(
            "CUDA not available. Unsloth 4-bit training expects an NVIDIA GPU on Linux/WSL.",
            file=__import__("sys").stderr,
        )
        return 1

    if not args.train.is_file():
        print(f"missing {args.train}; run scripts/prepare_sft_data.py first", file=__import__("sys").stderr)
        return 1

    train_ds = load_dataset("json", data_files=str(args.train), split="train")
    eval_ds = None
    if args.eval.is_file():
        eval_ds = load_dataset("json", data_files=str(args.eval), split="train")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        use_rslora=False,
        loftq_config=None,
    )

    def formatting_prompts_func(examples: dict) -> dict:
        texts = [
            tokenizer.apply_chat_template(
                convo,
                tokenize=False,
                add_generation_prompt=False,
            )
            for convo in examples["messages"]
        ]
        return {"text": texts}

    train_ds = train_ds.map(
        formatting_prompts_func,
        batched=True,
        remove_columns=["messages"],
    )
    if eval_ds is not None:
        eval_ds = eval_ds.map(
            formatting_prompts_func,
            batched=True,
            remove_columns=["messages"],
        )

    global_batch = max(1, args.batch_size * args.grad_accum)
    n_train = len(train_ds)
    steps_per_epoch = max(1, math.ceil(n_train / global_batch))
    total_update_steps = max(1, math.ceil(n_train * args.epochs / global_batch))
    step_interval = max(1, min(20, (steps_per_epoch + 2) // 3))
    warmup_steps = max(1, min(10, total_update_steps // 10))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fp16 = not torch.cuda.is_bf16_supported()
    bf16 = torch.cuda.is_bf16_supported()

    print(
        f"train_rows={n_train} global_batch={global_batch} "
        f"steps_per_epoch≈{steps_per_epoch} total_steps≈{total_update_steps} "
        f"warmup_steps={warmup_steps} save/eval_every={step_interval}",
        flush=True,
    )

    sft_args = SFTConfig(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=warmup_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=fp16,
        bf16=bf16,
        logging_steps=max(1, min(5, step_interval)),
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=step_interval if eval_ds is not None else None,
        save_strategy="steps",
        save_steps=step_interval,
        report_to="none",
        gradient_checkpointing=False,
        max_length=args.max_seq_length,
        packing=False,
        dataset_num_proc=1,
        eos_token=None,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(str(args.output_dir))
    print(f"Done. Adapter and tokenizer under {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
