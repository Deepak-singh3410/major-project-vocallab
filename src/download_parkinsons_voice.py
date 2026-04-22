from datasets import load_dataset

print("Downloading dataset...")

ds = load_dataset("birgermoell/Italian_Parkinsons_Voice_and_Speech")

print("Download complete!")

# Show dataset structure
print(ds)

# Show one sample
print("\nSample data:")
print(ds["train"][0])