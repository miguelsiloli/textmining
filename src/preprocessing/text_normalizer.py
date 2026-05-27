"""
Text Normalization Module for Financial Tweet Preprocessing

This module provides offline text normalization to reduce vocabulary sparsity
and improve model generalization by replacing specific entities with generic tokens.

Key Features:
- Ticker symbol normalization ($TSLA → [TICKER])
- Company name replacement (Tesla → [COMPANY])
- User mention normalization (@elonmusk → [USER])
- URL normalization (https://... → [URL])
- Optional numerical normalization (prices, percentages)

Expected Impact: +3-6% accuracy improvement through vocabulary reduction
"""

import re
import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pandas as pd


class TextNormalizer:
    """
    Normalizes financial tweets by replacing specific entities with generic tokens.
    
    This deterministic preprocessor should be applied to ALL data (train/val/test)
    before training to ensure consistent vocabulary across datasets.
    """
    
    def __init__(
        self,
        tickers: Optional[List[str]] = None,
        companies: Optional[List[str]] = None,
        normalize_prices: bool = False,
        normalize_percentages: bool = False,
        normalize_mentions: bool = True,
        normalize_urls: bool = True,
        vocab_analysis_path: Optional[str] = None
    ):
        """
        Initialize the text normalizer.
        
        Args:
            tickers: List of ticker symbols to replace (without $)
            companies: List of company names to replace
            normalize_prices: Whether to replace prices with [PRICE]
            normalize_percentages: Whether to replace percentages with [PERCENT]
            normalize_mentions: Whether to replace @mentions with [USER]
            normalize_urls: Whether to replace URLs with [URL]
            vocab_analysis_path: Path to vocabulary analysis results JSON
        """
        self.normalize_prices = normalize_prices
        self.normalize_percentages = normalize_percentages
        self.normalize_mentions = normalize_mentions
        self.normalize_urls = normalize_urls
        
        # Load from vocabulary analysis if provided
        if vocab_analysis_path:
            self._load_from_analysis(vocab_analysis_path)
        else:
            self.tickers = tickers or []
            self.companies = companies or []
        
        # Build regex patterns
        self._compile_patterns()
        
        # Statistics tracking
        self.stats = {
            'tickers_replaced': 0,
            'companies_replaced': 0,
            'mentions_replaced': 0,
            'urls_replaced': 0,
            'prices_replaced': 0,
            'percentages_replaced': 0
        }
    
    def _load_from_analysis(self, analysis_path: str):
        """Load ticker and company lists from vocabulary analysis results"""
        analysis_file = Path(analysis_path)
        
        if not analysis_file.exists():
            raise FileNotFoundError(f"Vocabulary analysis file not found: {analysis_path}")
        
        with open(analysis_file, 'r') as f:
            results = json.load(f)
        
        # Extract tickers (high confidence only for safety)
        self.tickers = [
            ticker for ticker, info in results.get('tickers', {}).items()
            if info.get('confidence') == 'high'  # Use only high-confidence tickers
        ]
        
        # Extract companies
        self.companies = list(results.get('companies', {}).keys())
        
        print(f"✓ Loaded {len(self.tickers)} tickers and {len(self.companies)} companies from analysis")
    
    def _compile_patterns(self):
        """Compile regex patterns for efficient matching"""
        
        # Ticker patterns: Match ANY $TICKER with 1-5 uppercase letters
        # This is more aggressive but catches all tickers including rare ones
        # Pattern: $[A-Z]{1,5} with word boundary
        self.ticker_pattern = re.compile(
            r'\$[A-Z]{1,5}\b',
            re.IGNORECASE
        )
        
        # Company name patterns
        if self.companies and len(self.companies) > 0:
            # Escape special characters and create pattern
            company_escaped = [re.escape(c) for c in self.companies]
            self.company_pattern = re.compile(
                r'\b(' + '|'.join(company_escaped) + r')\b',
                re.IGNORECASE
            )
        else:
            self.company_pattern = None
        
        # URL pattern
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        
        # Mention pattern
        self.mention_pattern = re.compile(r'@\w+')
        
        # Price pattern: $100, $1.5M, $2B
        self.price_pattern = re.compile(r'\$\d+\.?\d*[KMB]?')
        
        # Percentage pattern: 5%, 10.5%
        self.percentage_pattern = re.compile(r'\b\d+\.?\d*%')
    
    def normalize(self, text: str, track_stats: bool = True) -> str:
        """
        Normalize a single text sample.
        
        Args:
            text: Input text to normalize
            track_stats: Whether to track replacement statistics
            
        Returns:
            Normalized text with entities replaced by generic tokens
        """
        if pd.isna(text):
            return text
        
        original_text = str(text)
        
        # 1. Replace URLs (do this first to avoid matching URLs with tickers)
        if self.normalize_urls:
            url_count = len(self.url_pattern.findall(original_text))
            if url_count > 0 and track_stats:
                self.stats['urls_replaced'] += url_count
            original_text = self.url_pattern.sub('[URL]', original_text)
        
        # 2. Replace ticker symbols (both $TICKER and standalone)
        if self.ticker_pattern:
            def ticker_replacer(match):
                if track_stats:
                    self.stats['tickers_replaced'] += 1
                return '[TICKER]'
            
            original_text = self.ticker_pattern.sub(ticker_replacer, original_text)
        
        # 3. Replace company names
        if self.company_pattern:
            def company_replacer(match):
                if track_stats:
                    self.stats['companies_replaced'] += 1
                return '[COMPANY]'
            
            original_text = self.company_pattern.sub(company_replacer, original_text)
        
        # 4. Replace user mentions
        if self.normalize_mentions:
            mention_count = len(self.mention_pattern.findall(original_text))
            if mention_count > 0 and track_stats:
                self.stats['mentions_replaced'] += mention_count
            original_text = self.mention_pattern.sub('[USER]', original_text)
        
        # 5. Replace prices (optional)
        if self.normalize_prices:
            price_count = len(self.price_pattern.findall(original_text))
            if price_count > 0 and track_stats:
                self.stats['prices_replaced'] += price_count
            original_text = self.price_pattern.sub('[PRICE]', original_text)
        
        # 6. Replace percentages (optional)
        if self.normalize_percentages:
            pct_count = len(self.percentage_pattern.findall(original_text))
            if pct_count > 0 and track_stats:
                self.stats['percentages_replaced'] += pct_count
            original_text = self.percentage_pattern.sub('[PERCENT]', original_text)
        
        # Clean up multiple spaces
        original_text = re.sub(r'\s+', ' ', original_text).strip()
        
        return original_text
    
    def normalize_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = 'text',
        output_column: str = 'text_normalized',
        inplace: bool = False
    ) -> pd.DataFrame:
        """
        Normalize an entire DataFrame.
        
        Args:
            df: Input DataFrame
            text_column: Name of the column containing text
            output_column: Name of the column to store normalized text
            inplace: Whether to modify the DataFrame in place
            
        Returns:
            DataFrame with normalized text column added
        """
        if not inplace:
            df = df.copy()
        
        print(f"Normalizing {len(df)} samples...")
        
        # Reset stats
        self.stats = {k: 0 for k in self.stats}
        
        # Apply normalization
        df[output_column] = df[text_column].apply(self.normalize)
        
        print("✓ Normalization complete!")
        self._print_stats()
        
        return df
    
    def get_sample_comparisons(
        self,
        df: pd.DataFrame,
        n_samples: int = 5,
        text_column: str = 'text',
        normalized_column: str = 'text_normalized',
        seed: int = 42
    ) -> List[Tuple[str, str]]:
        """
        Get sample comparisons of original vs normalized text.
        
        Args:
            df: DataFrame with both original and normalized text
            n_samples: Number of samples to return
            text_column: Column with original text
            normalized_column: Column with normalized text
            seed: Random seed for sampling
            
        Returns:
            List of (original, normalized) tuples
        """
        # Sample rows where normalization made changes
        changed_mask = df[text_column] != df[normalized_column]
        changed_df = df[changed_mask]
        
        if len(changed_df) == 0:
            print("⚠ No changes detected. Check if normalization patterns match the data.")
            return []
        
        sample_df = changed_df.sample(min(n_samples, len(changed_df)), random_state=seed)
        
        comparisons = [
            (row[text_column], row[normalized_column])
            for _, row in sample_df.iterrows()
        ]
        
        return comparisons
    
    def _print_stats(self):
        """Print normalization statistics"""
        print("\n" + "="*80)
        print("NORMALIZATION STATISTICS")
        print("="*80)
        for entity_type, count in self.stats.items():
            if count > 0:
                print(f"  {entity_type.replace('_', ' ').title()}: {count:,}")
        print("="*80)
    
    def save_config(self, output_path: str):
        """Save normalizer configuration for reproducibility"""
        config = {
            'tickers': self.tickers,
            'companies': self.companies,
            'normalize_prices': self.normalize_prices,
            'normalize_percentages': self.normalize_percentages,
            'normalize_mentions': self.normalize_mentions,
            'normalize_urls': self.normalize_urls,
        }
        
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✓ Configuration saved to {output_path}")
    
    @classmethod
    def from_config(cls, config_path: str) -> 'TextNormalizer':
        """Load normalizer from saved configuration"""
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        return cls(**config)


def normalize_dataset(
    input_path: str,
    output_path: str,
    vocab_analysis_path: Optional[str] = None,
    normalize_prices: bool = True,  # Changed to True by default
    normalize_percentages: bool = True,  # Changed to True by default
    text_column: str = 'text',
    output_column: str = 'text_normalized'
) -> Tuple[pd.DataFrame, TextNormalizer]:
    """
    Convenience function to normalize a dataset and save it.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to save normalized CSV file
        vocab_analysis_path: Path to vocabulary analysis JSON
        normalize_prices: Whether to replace prices
        normalize_percentages: Whether to replace percentages
        text_column: Name of text column
        output_column: Name for normalized text column
        
    Returns:
        Tuple of (normalized DataFrame, TextNormalizer instance)
    """
    # Load data
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"✓ Loaded {len(df)} samples")
    
    # Initialize normalizer
    print("\nInitializing normalizer...")
    normalizer = TextNormalizer(
        vocab_analysis_path=vocab_analysis_path,
        normalize_prices=normalize_prices,
        normalize_percentages=normalize_percentages,
        normalize_mentions=True,
        normalize_urls=True
    )
    
    # Normalize
    print("\n" + "="*80)
    print("STARTING NORMALIZATION")
    print("="*80)
    df_normalized = normalizer.normalize_dataframe(
        df,
        text_column=text_column,
        output_column=output_column,
        inplace=False
    )
    
    # Save
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_normalized.to_csv(output_path, index=False)
    print(f"\n✓ Normalized dataset saved to {output_path}")
    
    return df_normalized, normalizer


def main():
    """Main execution for standalone usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Normalize financial tweet text')
    parser.add_argument('input', help='Input CSV file path')
    parser.add_argument('output', help='Output CSV file path')
    parser.add_argument('--vocab-analysis', help='Path to vocabulary analysis JSON')
    parser.add_argument('--prices', action='store_true', help='Normalize prices')
    parser.add_argument('--percentages', action='store_true', help='Normalize percentages')
    parser.add_argument('--text-col', default='text', help='Text column name')
    parser.add_argument('--samples', type=int, default=5, help='Number of sample comparisons to show')
    
    args = parser.parse_args()
    
    # Normalize dataset
    df_normalized, normalizer = normalize_dataset(
        input_path=args.input,
        output_path=args.output,
        vocab_analysis_path=args.vocab_analysis,
        normalize_prices=args.prices,
        normalize_percentages=args.percentages,
        text_column=args.text_col
    )
    
    # Show sample comparisons
    print("\n" + "="*80)
    print(f"SAMPLE COMPARISONS (showing {args.samples} examples)")
    print("="*80)
    
    comparisons = normalizer.get_sample_comparisons(
        df_normalized,
        n_samples=args.samples,
        text_column=args.text_col
    )
    
    for i, (original, normalized) in enumerate(comparisons, 1):
        print(f"\n[{i}] ORIGINAL:")
        print(f"    {original}")
        print(f"    NORMALIZED:")
        print(f"    {normalized}")
    
    # Save config
    config_path = Path(args.output).parent / 'normalizer_config.json'
    normalizer.save_config(str(config_path))


if __name__ == '__main__':
    main()
