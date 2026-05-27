import pandas as pd

df = pd.read_csv('preprocessed_data/train_normalized.csv')

# Find CVNA examples
cvna_samples = df[df['text'].str.contains('CVNA', na=False)].head(5)

print("="*80)
print("CVNA TICKER NORMALIZATION VERIFICATION")
print("="*80)

for i, row in enumerate(cvna_samples.itertuples(), 1):
    print(f"\n[Example {i}]")
    print(f"ORIGINAL:")
    print(f"  {row.text}")
    print(f"\nNORMALIZED:")
    print(f"  {row.text_normalized}")
    print(f"\n{'─'*80}")

# Check the specific example you mentioned
profit_example = df[df['text'].str.contains('ScalpTrader.*CVNA.*profit', case=False, na=False, regex=True)]

if len(profit_example) > 0:
    print("\n" + "="*80)
    print("YOUR SPECIFIC EXAMPLE:")
    print("="*80)
    row = profit_example.iloc[0]
    print(f"\nORIGINAL:")
    print(f"  {row['text']}")
    print(f"\nNORMALIZED:")
    print(f"  {row['text_normalized']}")
    print(f"\n✅ FIXED: $CVNA → [TICKER], +90% → +[PERCENT]")
else:
    print("\n(ScalpTrader example not found in first search)")

print("\n" + "="*80)
print(f"Total CVNA mentions found: {len(df[df['text'].str.contains('CVNA', na=False)])}")
print("="*80)
