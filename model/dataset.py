from datasets import load_dataset

ds = load_dataset("jtatman/stable-diffusion-prompts-stats-full-uncensored", split="train")
sfw_ds = ds.filter(lambda x: x["nsfw_score"] < 0.3 and x["nsfw_label"] == "SFW")

print(f"Filtered SFW size: {len(sfw_ds)}")
