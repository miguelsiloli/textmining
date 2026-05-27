from preprocessing.text_normalizer import TextNormalizer

# Initialize with aggressive pattern matching
normalizer = TextNormalizer(
    tickers=[],  # Empty list - will use pattern matching
    companies=['Tesla', 'Apple', 'Amazon', 'Boeing'],
    normalize_prices=True,
    normalize_percentages=True,
    normalize_mentions=True,
    normalize_urls=True
)

# Test cases
test_texts = [
    "$CVNA is up +90%, worth $26bn",
    "$TSLA CEO @elonmusk says buy",
    "Apple stock rising 5% today",
    "@user check https://test.com for details",
    "On 3/24 ScalpTrader took profits on $CVNA for +90%, or +22 pts.",
    "$SPY $QQQ $AAPL all green today",
    "Tesla announces $150M investment, stock up 12%"
]

print("="*80)
print("IMPROVED NORMALIZATION TEST")
print("="*80)

for i, text in enumerate(test_texts, 1):
    normalized = normalizer.normalize(text)
    print(f"\n[{i}] ORIGINAL:")
    print(f"    {text}")
    print(f"    NORMALIZED:")
    print(f"    {normalized}")

print("\n" + "="*80)
print("Statistics:")
print(f"  Tickers replaced: {normalizer.stats['tickers_replaced']}")
print(f"  Companies replaced: {normalizer.stats['companies_replaced']}")
print(f"  Mentions replaced: {normalizer.stats['mentions_replaced']}")
print(f"  URLs replaced: {normalizer.stats['urls_replaced']}")
print(f"  Prices replaced: {normalizer.stats['prices_replaced']}")
print(f"  Percentages replaced: {normalizer.stats['percentages_replaced']}")
print("="*80)
