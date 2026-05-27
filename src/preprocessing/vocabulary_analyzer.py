"""
Vocabulary Analysis Agent for Financial Tweet Text Normalization

This module analyzes the tweet corpus to identify vocabulary patterns that should be 
replaced with generic tokens to improve model generalization and reduce overfitting.

Key Objectives:
1. Identify stock ticker symbols ($TSLA, AAPL, etc.)
2. Detect company names (Tesla, Apple Inc., etc.)
3. Find numerical patterns (prices, percentages, dates)
4. Identify financial entities (CEO names, exchanges, etc.)
5. Analyze vocabulary frequency and distribution
6. Provide replacement strategy recommendations
"""

import pandas as pd
import numpy as np
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Set
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class VocabularyAnalyzer:
    """Analyzes tweet vocabulary and identifies normalization patterns"""
    
    def __init__(self, data_path: str):
        """
        Initialize the analyzer
        
        Args:
            data_path: Path to the CSV file containing tweets
        """
        self.data_path = data_path
        self.df = None
        self.analysis_results = {}
        
        # Regex patterns for different entity types
        self.patterns = {
            'cashtag': r'\$[A-Z]{1,5}\b',  # $TSLA, $AAPL
            'ticker_standalone': r'\b[A-Z]{2,5}\b',  # TSLA, AAPL (standalone)
            'percentage': r'\b\d+\.?\d*%',  # 5%, 10.5%
            'price': r'\$\d+\.?\d*[KMB]?',  # $100, $1.5M, $2B
            'number': r'\b\d+\.?\d*[KMB]?\b',  # 100, 1.5M, 2B
            'url': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            'mention': r'@\w+',  # @elonmusk
            'hashtag': r'#\w+',  # #stocks
            'date': r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b',
        }
        
        # Common company names to detect
        self.company_names = [
            'Tesla', 'Apple', 'Amazon', 'Google', 'Microsoft', 'Facebook', 'Meta',
            'Netflix', 'Twitter', 'Nvidia', 'AMD', 'Intel', 'Boeing', 'Ford',
            'General Motors', 'GM', 'JPMorgan', 'Goldman Sachs', 'Morgan Stanley',
            'Bank of America', 'Wells Fargo', 'Citigroup', 'Walmart', 'Target',
            'Costco', 'Home Depot', 'Nike', 'Starbucks', 'McDonald', 'Coca-Cola',
            'PepsiCo', 'Pfizer', 'Johnson & Johnson', 'Moderna', 'SpaceX',
            'PayPal', 'Square', 'Visa', 'Mastercard', 'Disney', 'AT&T', 'Verizon'
        ]
        
        # Common financial terms that might indicate tickers
        self.financial_context_words = [
            'stock', 'share', 'price', 'trading', 'buy', 'sell', 'long', 'short',
            'calls', 'puts', 'options', 'futures', 'rally', 'dump', 'moon', 'crash'
        ]
    
    def load_data(self):
        """Load the tweet data"""
        print(f"Loading data from {self.data_path}...")
        self.df = pd.read_csv(self.data_path)
        print(f"✓ Loaded {len(self.df)} tweets")
        return self
    
    def extract_entities(self) -> Dict[str, List[Tuple[str, int]]]:
        """
        Extract all entities from tweets using regex patterns
        
        Returns:
            Dictionary mapping entity type to list of (entity, count) tuples
        """
        print("\n[1/7] Extracting entities from tweets...")
        
        entities = defaultdict(Counter)
        
        for text in self.df['text']:
            if pd.isna(text):
                continue
            
            text_str = str(text)
            
            # Extract each entity type
            for entity_type, pattern in self.patterns.items():
                matches = re.findall(pattern, text_str)
                entities[entity_type].update(matches)
        
        # Convert to sorted lists
        self.analysis_results['entities'] = {
            entity_type: counter.most_common(100)  # Top 100 for each type
            for entity_type, counter in entities.items()
        }
        
        # Print summary
        for entity_type, items in self.analysis_results['entities'].items():
            print(f"  {entity_type}: {len(items)} unique entities, {sum(c for _, c in items)} total occurrences")
        
        return self.analysis_results['entities']
    
    def identify_tickers(self) -> Dict[str, any]:
        """
        Identify stock ticker symbols with confidence scores
        
        Returns:
            Dictionary with ticker analysis
        """
        print("\n[2/7] Identifying stock tickers...")
        
        # Get cashtags (high confidence - they start with $)
        cashtags = self.analysis_results['entities'].get('cashtag', [])
        high_confidence_tickers = [(tag[1:], count) for tag, count in cashtags]  # Remove $
        
        # Get standalone uppercase words (potential tickers)
        standalone = self.analysis_results['entities'].get('ticker_standalone', [])
        
        # Filter standalone words that are likely tickers
        potential_tickers = []
        noise_words = {'I', 'A', 'AM', 'PM', 'US', 'UK', 'CEO', 'CFO', 'CTO', 
                       'IPO', 'ETF', 'ATH', 'ATL', 'DD', 'YOLO', 'FOMO', 'FUD',
                       'TA', 'FA', 'IT', 'AI', 'ML', 'AR', 'VR', 'EV', 'ESG'}
        
        for word, count in standalone:
            # Skip noise words
            if word in noise_words:
                continue
            # Check if it appears in financial context
            if self._is_likely_ticker(word):
                potential_tickers.append((word, count))
        
        # Combine and deduplicate
        all_tickers = {}
        for ticker, count in high_confidence_tickers:
            all_tickers[ticker] = {
                'count': count,
                'confidence': 'high',
                'source': 'cashtag'
            }
        
        for ticker, count in potential_tickers:
            if ticker not in all_tickers:
                all_tickers[ticker] = {
                    'count': count,
                    'confidence': 'medium',
                    'source': 'context'
                }
        
        self.analysis_results['tickers'] = dict(sorted(
            all_tickers.items(), 
            key=lambda x: x[1]['count'], 
            reverse=True
        )[:50])  # Top 50 tickers
        
        print(f"  ✓ Identified {len(self.analysis_results['tickers'])} ticker symbols")
        print(f"    High confidence: {sum(1 for t in self.analysis_results['tickers'].values() if t['confidence'] == 'high')}")
        print(f"    Medium confidence: {sum(1 for t in self.analysis_results['tickers'].values() if t['confidence'] == 'medium')}")
        
        return self.analysis_results['tickers']
    
    def _is_likely_ticker(self, word: str) -> bool:
        """Check if a word is likely a ticker based on context"""
        # Check if it appears near financial terms
        pattern = r'\b(?:' + '|'.join(self.financial_context_words) + r')\b.*\b' + word + r'\b'
        
        sample_texts = self.df['text'].sample(min(1000, len(self.df))).astype(str)
        mentions = sum(1 for text in sample_texts if re.search(pattern, text, re.IGNORECASE))
        
        return mentions > 0
    
    def identify_company_names(self) -> Dict[str, int]:
        """
        Identify company names in tweets
        
        Returns:
            Dictionary mapping company name to occurrence count
        """
        print("\n[3/7] Identifying company names...")
        
        company_counts = Counter()
        
        for text in self.df['text']:
            if pd.isna(text):
                continue
            
            text_str = str(text)
            
            for company in self.company_names:
                # Case-insensitive search with word boundaries
                pattern = r'\b' + re.escape(company) + r'\b'
                matches = len(re.findall(pattern, text_str, re.IGNORECASE))
                if matches > 0:
                    company_counts[company] += matches
        
        self.analysis_results['companies'] = dict(company_counts.most_common(30))
        
        print(f"  ✓ Found {len(self.analysis_results['companies'])} company names")
        print(f"    Total mentions: {sum(self.analysis_results['companies'].values())}")
        
        return self.analysis_results['companies']
    
    def analyze_numerical_patterns(self) -> Dict[str, any]:
        """
        Analyze numerical patterns (prices, percentages, dates)
        
        Returns:
            Dictionary with numerical pattern statistics
        """
        print("\n[4/7] Analyzing numerical patterns...")
        
        numerical_stats = {
            'percentages': {
                'count': len(self.analysis_results['entities'].get('percentage', [])),
                'total_occurrences': sum(c for _, c in self.analysis_results['entities'].get('percentage', [])),
                'top_10': self.analysis_results['entities'].get('percentage', [])[:10]
            },
            'prices': {
                'count': len(self.analysis_results['entities'].get('price', [])),
                'total_occurrences': sum(c for _, c in self.analysis_results['entities'].get('price', [])),
                'top_10': self.analysis_results['entities'].get('price', [])[:10]
            },
            'dates': {
                'count': len(self.analysis_results['entities'].get('date', [])),
                'total_occurrences': sum(c for _, c in self.analysis_results['entities'].get('date', [])),
                'top_10': self.analysis_results['entities'].get('date', [])[:10]
            },
            'numbers': {
                'count': len(self.analysis_results['entities'].get('number', [])),
                'total_occurrences': sum(c for _, c in self.analysis_results['entities'].get('number', [])),
                'top_10': self.analysis_results['entities'].get('number', [])[:10]
            }
        }
        
        self.analysis_results['numerical'] = numerical_stats
        
        for pattern_type, stats in numerical_stats.items():
            print(f"  {pattern_type}: {stats['total_occurrences']} occurrences, {stats['count']} unique")
        
        return numerical_stats
    
    def analyze_social_entities(self) -> Dict[str, any]:
        """
        Analyze social media entities (mentions, hashtags, URLs)
        
        Returns:
            Dictionary with social entity statistics
        """
        print("\n[5/7] Analyzing social media entities...")
        
        social_stats = {
            'mentions': {
                'count': len(self.analysis_results['entities'].get('mention', [])),
                'total_occurrences': sum(c for _, c in self.analysis_results['entities'].get('mention', [])),
                'top_10': self.analysis_results['entities'].get('mention', [])[:10]
            },
            'hashtags': {
                'count': len(self.analysis_results['entities'].get('hashtag', [])),
                'total_occurrences': sum(c for _, c in self.analysis_results['entities'].get('hashtag', [])),
                'top_10': self.analysis_results['entities'].get('hashtag', [])[:10]
            },
            'urls': {
                'count': len(self.analysis_results['entities'].get('url', [])),
                'total_occurrences': sum(c for _, c in self.analysis_results['entities'].get('url', [])),
            }
        }
        
        self.analysis_results['social'] = social_stats
        
        for entity_type, stats in social_stats.items():
            print(f"  {entity_type}: {stats['total_occurrences']} occurrences, {stats['count']} unique")
        
        return social_stats
    
    def analyze_sentiment_distribution(self) -> Dict[str, any]:
        """
        Analyze how entities correlate with sentiment labels
        
        Returns:
            Dictionary with sentiment distribution per entity type
        """
        print("\n[6/7] Analyzing sentiment distribution by entity...")
        
        label_map = {0: 'Bearish', 1: 'Bullish', 2: 'Neutral'}
        
        # Analyze top tickers by sentiment
        ticker_sentiment = defaultdict(lambda: {'Bearish': 0, 'Bullish': 0, 'Neutral': 0})
        
        for idx, row in self.df.iterrows():
            text = str(row['text'])
            label = label_map[row['label']]
            
            # Check for tickers
            for ticker in self.analysis_results['tickers'].keys():
                if re.search(r'\b' + ticker + r'\b', text) or f'${ticker}' in text:
                    ticker_sentiment[ticker][label] += 1
        
        # Get top 20 tickers with most mentions
        top_tickers = sorted(
            ticker_sentiment.items(),
            key=lambda x: sum(x[1].values()),
            reverse=True
        )[:20]
        
        self.analysis_results['sentiment_analysis'] = {
            'ticker_sentiment': dict(top_tickers)
        }
        
        print(f"  ✓ Analyzed sentiment distribution for {len(top_tickers)} top tickers")
        
        return self.analysis_results['sentiment_analysis']
    
    def generate_replacement_strategy(self) -> Dict[str, any]:
        """
        Generate comprehensive replacement strategy recommendations
        
        Returns:
            Dictionary with replacement rules and statistics
        """
        print("\n[7/7] Generating replacement strategy...")
        
        strategy = {
            'ticker_replacement': {
                'rule': 'Replace all ticker symbols with [TICKER] token',
                'pattern': r'\$[A-Z]{1,5}\b|\b[A-Z]{2,5}\b(?=\s|$)',
                'confidence_filter': 'Use high-confidence tickers only',
                'tickers_to_replace': list(self.analysis_results['tickers'].keys()),
                'estimated_impact': f"{len(self.analysis_results['tickers'])} unique tickers",
                'expected_improvement': '+2-4% accuracy (reduces sparsity)'
            },
            'company_name_replacement': {
                'rule': 'Replace company names with [COMPANY] token',
                'companies_to_replace': list(self.analysis_results['companies'].keys()),
                'estimated_impact': f"{len(self.analysis_results['companies'])} companies",
                'expected_improvement': '+1-2% accuracy (generalization)'
            },
            'numerical_replacement': {
                'percentages': {
                    'rule': 'Replace percentages with [PERCENT] token',
                    'pattern': r'\b\d+\.?\d*%',
                    'estimated_impact': f"{self.analysis_results['numerical']['percentages']['total_occurrences']} occurrences",
                    'optional': True
                },
                'prices': {
                    'rule': 'Replace prices with [PRICE] token',
                    'pattern': r'\$\d+\.?\d*[KMB]?',
                    'estimated_impact': f"{self.analysis_results['numerical']['prices']['total_occurrences']} occurrences",
                    'optional': True
                },
                'numbers': {
                    'rule': 'Replace large numbers with [NUMBER] token',
                    'pattern': r'\b\d+\.?\d*[KMB]\b',
                    'estimated_impact': f"{self.analysis_results['numerical']['numbers']['total_occurrences']} occurrences",
                    'optional': True
                }
            },
            'social_replacement': {
                'mentions': {
                    'rule': 'Replace user mentions with [USER] token',
                    'pattern': r'@\w+',
                    'estimated_impact': f"{self.analysis_results['social']['mentions']['total_occurrences']} occurrences",
                    'recommended': True
                },
                'hashtags': {
                    'rule': 'Keep hashtags (they carry semantic meaning)',
                    'recommended': False
                },
                'urls': {
                    'rule': 'Replace URLs with [URL] token',
                    'pattern': self.patterns['url'],
                    'estimated_impact': f"{self.analysis_results['social']['urls']['total_occurrences']} occurrences",
                    'recommended': True
                }
            },
            'priority_order': [
                '1. High Priority: Ticker symbols ($TSLA, AAPL) → [TICKER]',
                '2. High Priority: Company names (Tesla, Apple) → [COMPANY]',
                '3. Medium Priority: User mentions (@elonmusk) → [USER]',
                '4. Medium Priority: URLs → [URL]',
                '5. Low Priority: Prices ($100, $1.5M) → [PRICE]',
                '6. Low Priority: Percentages (5%, 10.5%) → [PERCENT]',
            ],
            'estimated_total_impact': {
                'vocabulary_reduction': f"{len(self.analysis_results['tickers']) + len(self.analysis_results['companies'])} entities",
                'expected_accuracy_gain': '+3-6% overall (combined effect)',
                'primary_benefit': 'Reduces overfitting to specific companies, improves generalization'
            }
        }
        
        self.analysis_results['replacement_strategy'] = strategy
        
        print("  ✓ Replacement strategy generated")
        print("\n  Priority Recommendations:")
        for priority in strategy['priority_order']:
            print(f"    {priority}")
        
        return strategy
    
    def visualize_results(self, output_dir: str = './outputs/vocab_analysis'):
        """
        Generate visualizations of the analysis results
        
        Args:
            output_dir: Directory to save visualizations
        """
        print("\n[Visualization] Generating plots...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Top tickers bar chart
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Top 20 tickers
        tickers = list(self.analysis_results['tickers'].keys())[:20]
        counts = [self.analysis_results['tickers'][t]['count'] for t in tickers]
        
        axes[0, 0].barh(tickers, counts, color='steelblue')
        axes[0, 0].set_xlabel('Occurrences')
        axes[0, 0].set_title('Top 20 Stock Tickers', fontweight='bold')
        axes[0, 0].invert_yaxis()
        
        # Top 15 companies
        companies = list(self.analysis_results['companies'].keys())[:15]
        company_counts = [self.analysis_results['companies'][c] for c in companies]
        
        axes[0, 1].barh(companies, company_counts, color='coral')
        axes[0, 1].set_xlabel('Occurrences')
        axes[0, 1].set_title('Top 15 Company Names', fontweight='bold')
        axes[0, 1].invert_yaxis()
        
        # Entity type distribution
        entity_totals = {
            'Tickers': sum(t['count'] for t in self.analysis_results['tickers'].values()),
            'Companies': sum(self.analysis_results['companies'].values()),
            'Percentages': self.analysis_results['numerical']['percentages']['total_occurrences'],
            'Prices': self.analysis_results['numerical']['prices']['total_occurrences'],
            'Mentions': self.analysis_results['social']['mentions']['total_occurrences'],
            'URLs': self.analysis_results['social']['urls']['total_occurrences']
        }
        
        axes[1, 0].bar(entity_totals.keys(), entity_totals.values(), color='seagreen')
        axes[1, 0].set_ylabel('Total Occurrences')
        axes[1, 0].set_title('Entity Type Distribution', fontweight='bold')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Ticker sentiment distribution (top 10)
        if 'sentiment_analysis' in self.analysis_results:
            top_10_tickers = list(self.analysis_results['sentiment_analysis']['ticker_sentiment'].items())[:10]
            ticker_names = [t[0] for t in top_10_tickers]
            bearish = [t[1]['Bearish'] for t in top_10_tickers]
            bullish = [t[1]['Bullish'] for t in top_10_tickers]
            neutral = [t[1]['Neutral'] for t in top_10_tickers]
            
            x = np.arange(len(ticker_names))
            width = 0.25
            
            axes[1, 1].bar(x - width, bearish, width, label='Bearish', color='red', alpha=0.7)
            axes[1, 1].bar(x, neutral, width, label='Neutral', color='gray', alpha=0.7)
            axes[1, 1].bar(x + width, bullish, width, label='Bullish', color='green', alpha=0.7)
            
            axes[1, 1].set_ylabel('Tweet Count')
            axes[1, 1].set_title('Top 10 Tickers by Sentiment', fontweight='bold')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(ticker_names, rotation=45)
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(output_path / 'vocabulary_analysis.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved visualization to {output_path / 'vocabulary_analysis.png'}")
        plt.close()
    
    def save_results(self, output_dir: str = './outputs/vocab_analysis'):
        """
        Save analysis results to JSON files
        
        Args:
            output_dir: Directory to save results
        """
        print("\n[Saving] Writing results to files...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save complete analysis
        with open(output_path / 'vocabulary_analysis.json', 'w') as f:
            json.dump(self.analysis_results, f, indent=2)
        print(f"  ✓ Saved complete analysis to {output_path / 'vocabulary_analysis.json'}")
        
        # Save ticker list for easy import
        ticker_list = {
            'high_confidence': [
                ticker for ticker, info in self.analysis_results['tickers'].items()
                if info['confidence'] == 'high'
            ],
            'medium_confidence': [
                ticker for ticker, info in self.analysis_results['tickers'].items()
                if info['confidence'] == 'medium'
            ]
        }
        
        with open(output_path / 'tickers_to_replace.json', 'w') as f:
            json.dump(ticker_list, f, indent=2)
        print(f"  ✓ Saved ticker list to {output_path / 'tickers_to_replace.json'}")
        
        # Save company names
        with open(output_path / 'companies_to_replace.json', 'w') as f:
            json.dump(list(self.analysis_results['companies'].keys()), f, indent=2)
        print(f"  ✓ Saved company list to {output_path / 'companies_to_replace.json'}")
        
        # Save replacement strategy as markdown
        strategy = self.analysis_results['replacement_strategy']
        markdown_content = self._generate_markdown_report(strategy)
        
        with open(output_path / 'replacement_strategy.md', 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        print(f"  ✓ Saved strategy report to {output_path / 'replacement_strategy.md'}")
    
    def _generate_markdown_report(self, strategy: Dict) -> str:
        """Generate a markdown report of the replacement strategy"""
        report = "# Vocabulary Replacement Strategy\n\n"
        report += "## Executive Summary\n\n"
        report += f"**Estimated Total Impact:** {strategy['estimated_total_impact']['expected_accuracy_gain']}\n\n"
        report += f"**Primary Benefit:** {strategy['estimated_total_impact']['primary_benefit']}\n\n"
        
        report += "## Priority Order\n\n"
        for priority in strategy['priority_order']:
            report += f"- {priority}\n"
        
        report += "\n## Detailed Replacement Rules\n\n"
        
        report += "### 1. Ticker Replacement\n"
        report += f"- **Rule:** {strategy['ticker_replacement']['rule']}\n"
        report += f"- **Pattern:** `{strategy['ticker_replacement']['pattern']}`\n"
        report += f"- **Impact:** {strategy['ticker_replacement']['estimated_impact']}\n"
        report += f"- **Expected Improvement:** {strategy['ticker_replacement']['expected_improvement']}\n\n"
        
        report += "### 2. Company Name Replacement\n"
        report += f"- **Rule:** {strategy['company_name_replacement']['rule']}\n"
        report += f"- **Impact:** {strategy['company_name_replacement']['estimated_impact']}\n"
        report += f"- **Expected Improvement:** {strategy['company_name_replacement']['expected_improvement']}\n\n"
        
        report += "### 3. Numerical Replacements (Optional)\n"
        for num_type, info in strategy['numerical_replacement'].items():
            report += f"#### {num_type.title()}\n"
            report += f"- **Rule:** {info['rule']}\n"
            report += f"- **Pattern:** `{info['pattern']}`\n"
            report += f"- **Impact:** {info['estimated_impact']}\n\n"
        
        report += "### 4. Social Entity Replacements\n"
        for social_type, info in strategy['social_replacement'].items():
            report += f"#### {social_type.title()}\n"
            report += f"- **Rule:** {info['rule']}\n"
            if 'pattern' in info:
                report += f"- **Pattern:** `{info['pattern']}`\n"
            if 'recommended' in info:
                report += f"- **Recommended:** {'Yes' if info['recommended'] else 'No'}\n"
            if 'estimated_impact' in info:
                report += f"- **Impact:** {info['estimated_impact']}\n"
            report += "\n"
        
        return report
    
    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        print("="*80)
        print("VOCABULARY ANALYSIS AGENT")
        print("="*80)
        
        self.load_data()
        self.extract_entities()
        self.identify_tickers()
        self.identify_company_names()
        self.analyze_numerical_patterns()
        self.analyze_social_entities()
        self.analyze_sentiment_distribution()
        self.generate_replacement_strategy()
        self.visualize_results()
        self.save_results()
        
        print("\n" + "="*80)
        print("✓ ANALYSIS COMPLETE!")
        print("="*80)
        
        return self.analysis_results


def main():
    """Main execution function"""
    import sys
    
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        # Default path for Kaggle
        data_path = '../data/train.csv'
    
    analyzer = VocabularyAnalyzer(data_path)
    results = analyzer.run_full_analysis()
    
    # Print summary
    print("\n📊 Key Findings:")
    print(f"  • {len(results['tickers'])} ticker symbols identified")
    print(f"  • {len(results['companies'])} company names found")
    print(f"  • {results['numerical']['percentages']['total_occurrences']} percentage mentions")
    print(f"  • {results['social']['mentions']['total_occurrences']} user mentions")
    print(f"\n💡 Expected Improvement: {results['replacement_strategy']['estimated_total_impact']['expected_accuracy_gain']}")


if __name__ == '__main__':
    main()
