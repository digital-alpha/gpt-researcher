from typing import Dict, List, Optional, Tuple
import numpy as np
from langchain_aws import BedrockEmbeddings
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams, PointStruct
import logging
import uuid
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class ReportTypeClassifier:
    def __init__(
        self,
        qdrant_url: str = "localhost",
        qdrant_api_key: str = "",
        collection_name: str = "report_types",
        embedding_model_id: str = "amazon.titan-embed-text-v2:0",
        region_name: str = "us-east-1",
        force_recreate: bool = False
    ):
        """
        force_recreate: Whether to recreate the collection if it exists
        """
        self.collection_name = collection_name
        self.force_recreate = force_recreate
        
        self.qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        
        self.embeddings_model = BedrockEmbeddings(
            region_name=region_name,
            model_id=embedding_model_id
        )
        
        self._setup_collection()
    
    def _get_report_data(self) -> List[Dict]:
        """Define the report types and their characteristic phrases."""
        return [
            {
                "report_type": "FinancialAnalysis",
                "phrases": [
                    "analyze the quarterly earnings and revenue trends",
                    "break down the financial performance and profit margins",
                    "examine the balance sheet and cash flow statements",
                    "evaluate the dividend yield and return on investment",
                    "assess the debt-to-equity ratio and market capitalization",
                    "review the income statement and asset valuation",
                    "calculate financial ratios and investment returns"
                ]
            },
            {
                "report_type": "TechnicalAnalysis",
                "phrases": [
                    "analyze the chart patterns and price movements",
                    "examine the candlestick formations and trading signals",
                    "identify support and resistance levels in the market",
                    "evaluate moving averages and momentum indicators",
                    "assess RSI levels and MACD signal strength",
                    "analyze Bollinger bands and volume patterns",
                    "identify breakout patterns and trend reversals"
                ]
            },
            {
                "report_type": "MarketResearch",
                "phrases": [
                    "research the target market size and growth potential",
                    "analyze consumer behavior and buying patterns",
                    "examine the competitive landscape and market share",
                    "study customer demographics and segmentation",
                    "evaluate brand positioning and market penetration",
                    "forecast demand trends and market opportunities",
                    "assess consumer preferences and purchasing decisions"
                ]
            },
            {
                "report_type": "CompanyProfile",
                "phrases": [
                    "profile the company leadership and management team",
                    "examine the corporate structure and business model",
                    "research the company history and organizational development",
                    "analyze the board of directors and key executives",
                    "study the corporate governance and company culture",
                    "review the mission statement and strategic vision",
                    "assess the organizational hierarchy and reporting structure"
                ]
            },
            {
                "report_type": "IndustryAnalysis",
                "phrases": [
                    "analyze the industry trends and sector dynamics",
                    "examine regulatory requirements and compliance standards",
                    "study the supply chain and value chain structure",
                    "assess technological disruption and innovation impact",
                    "evaluate market leaders and emerging competitors",
                    "analyze industry challenges and growth drivers",
                    "examine the competitive environment and market forces"
                ]
            },
            {
                "report_type": "ComparativeStudy",
                "phrases": [
                    "compare the features and capabilities side by side",
                    "evaluate the pros and cons of different options",
                    "benchmark performance across multiple alternatives",
                    "analyze the competitive advantages and disadvantages",
                    "assess the cost-benefit trade-offs between choices",
                    "rank options based on specific criteria",
                    "determine the best alternative among competitors"
                ]
            },
            {
                "report_type": "TrendForecast",
                "phrases": [
                    "predict future market trends and developments",
                    "forecast growth patterns and emerging opportunities",
                    "analyze long-term trajectory and cyclical patterns",
                    "project future scenarios and market evolution",
                    "identify emerging trends and disruptive changes",
                    "estimate future demand and market conditions",
                    "anticipate industry shifts and technological advances"
                ]
            },
            {
                "report_type": "RiskAssessment",
                "phrases": [
                    "identify potential risks and threat scenarios",
                    "assess vulnerabilities and exposure levels",
                    "evaluate the likelihood and impact of risks",
                    "analyze security threats and operational dangers",
                    "examine downside scenarios and worst-case outcomes",
                    "develop risk mitigation and contingency strategies",
                    "calculate risk tolerance and acceptable exposure"
                ]
            },
            {
                "report_type": "OpportunityMapping",
                "phrases": [
                    "identify growth opportunities and investment prospects",
                    "discover untapped markets and expansion possibilities",
                    "explore partnership opportunities and strategic alliances",
                    "find new revenue streams and business opportunities",
                    "assess market gaps and unmet customer needs",
                    "evaluate first-mover advantages and competitive positioning",
                    "analyze emerging markets and innovation potential"
                ]
            },
            {
                "report_type": "PerformanceMetrics",
                "phrases": [
                    "track key performance indicators and success metrics",
                    "measure productivity and efficiency levels",
                    "monitor progress against targets and benchmarks",
                    "evaluate achievement and outcome measurements",
                    "create dashboards for performance tracking",
                    "assess team performance and individual contributions",
                    "analyze results and compare against industry standards"
                ]
            },
            {
                "report_type": "StrategicPlanning",
                "phrases": [
                    "develop a strategic roadmap and implementation plan",
                    "define long-term objectives and business strategy",
                    "create action plans and milestone schedules",
                    "establish strategic priorities and resource allocation",
                    "formulate vision statements and strategic initiatives",
                    "design implementation frameworks and execution plans",
                    "align goals with strategic direction and market position"
                ]
            },
            {
                "report_type": "ResourceCompilation",
                "phrases": [
                    "compile a comprehensive list of useful resources",
                    "gather relevant tools and reference materials",
                    "create a directory of helpful websites and databases",
                    "collect educational materials and learning resources",
                    "organize recommended reading and research sources",
                    "assemble a toolkit of practical resources",
                    "curate a bibliography of relevant publications"
                ]
            },
            {
                "report_type": "NewsDigest",
                "phrases": [
                    "summarize the latest news and current developments",
                    "compile recent updates and breaking announcements",
                    "digest current events and media coverage",
                    "review press releases and industry announcements",
                    "analyze headline trends and news patterns",
                    "track recent developments and market updates",
                    "monitor current affairs and regulatory changes"
                ]
            },
            {
                "report_type": "HowToGuide",
                "phrases": [
                    "provide step-by-step instructions and guidance",
                    "create a beginner tutorial and learning pathway",
                    "develop implementation guidelines and best practices",
                    "design a practical walkthrough and training manual",
                    "establish procedures and operational guidelines",
                    "create educational content and instructional materials",
                    "develop a comprehensive how-to manual"
                ]
            },
            {
                "report_type": "FAQCollection",
                "phrases": [
                    "address frequently asked questions and common concerns",
                    "provide answers to typical user inquiries",
                    "create troubleshooting guides and problem solutions",
                    "compile help documentation and support materials",
                    "answer common questions and clarify confusion",
                    "develop user support and assistance resources",
                    "create a knowledge base of common issues"
                ]
            },
            {
                "report_type": "ScientificStudy",
                "phrases": [
                    "conduct empirical research and statistical analysis",
                    "design controlled experiments and hypothesis testing",
                    "perform peer-reviewed academic research",
                    "analyze experimental data and research findings",
                    "develop research methodology and data collection protocols",
                    "conduct literature reviews and scientific investigations",
                    "publish research conclusions and academic findings"
                ]
            },
            {
                "report_type": "ExecutiveBrief",
                "phrases": [
                    "provide a concise executive summary and key highlights",
                    "create a high-level overview and main findings",
                    "deliver a quick briefing and essential points",
                    "summarize critical information for decision makers",
                    "present condensed analysis and strategic insights",
                    "offer executive-level synopsis and recommendations",
                    "provide senior leadership with essential information"
                ]
            },
            {
                "report_type": "DetailedInvestigation",
                "phrases": [
                    "conduct a comprehensive and thorough examination",
                    "perform an in-depth analysis and extensive research",
                    "execute a detailed investigation and complete assessment",
                    "carry out exhaustive review and meticulous study",
                    "perform deep-dive research and comprehensive evaluation",
                    "conduct full-scale analysis and complete investigation",
                    "execute thorough examination and detailed exploration"
                ]
            }
        ]
    
    def _setup_collection(self):
        """Setup Qdrant collection and populate with embeddings if needed."""
        try:
            collections = self.qdrant_client.get_collections().collections
            collection_exists = any(col.name == self.collection_name for col in collections)
            # print(collection_exists)
            
            if collection_exists and self.force_recreate:
                logger.info(f"Deleting existing collection: {self.collection_name}")
                self.qdrant_client.delete_collection(self.collection_name)
                collection_exists = False
            
            if not collection_exists:
                logger.info(f"Creating collection: {self.collection_name}")
                # Amazon Titan Text v2 -> 1024
                sample_embedding = self.embeddings_model.embed_query("sample text")
                vector_size = len(sample_embedding)
                
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE
                    )
                )
                
                self._populate_collection()
            else:
                logger.info(f"Using existing collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Error setting up collection: {e}")
            raise
    
    def _populate_collection(self):
        """Populate Qdrant collection with phrase embeddings."""
        logger.info("Populating collection with phrase embeddings...")
        
        points = []
        report_data = self._get_report_data()
        
        for report_info in report_data:
            report_type = report_info["report_type"]
            phrases = report_info["phrases"]
            
            phrase_embeddings = self.embeddings_model.embed_documents(phrases)
            for i, (phrase, embedding) in enumerate(zip(phrases, phrase_embeddings)):
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "report_type": report_type,
                        "phrase": phrase,
                        "phrase_index": i
                    }
                )
                points.append(point)
        
        #batch insert points
        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        logger.info(f"Successfully inserted {len(points)} phrase embeddings")
    
    def classify(
        self,
        user_query: str,
        top_k: int = 10,
        score_threshold: float = 0.0
    ) -> str:
        """
        Classify user query to determine the best report type using weighted scoring.
        
        Args:
            user_query: The user's query text
            top_k: Number of top similar phrases to retrieve
            score_threshold: Minimum similarity score threshold
            
        Returns:
            The predicted report type
        """
        try:
            query_embedding = self.embeddings_model.embed_query(user_query)
            
            search_results = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=top_k,
                score_threshold=score_threshold
            )
            
            if not search_results or not hasattr(search_results, 'points') or not search_results.points:
                logger.warning("No similar phrases found above threshold")
                return "DetailedInvestigation"
            
            report_scores = {}
            
            for result in search_results.points:
                report_type = result.payload["report_type"]
                
                if report_type not in report_scores:
                    report_scores[report_type] = []
    
                report_scores[report_type].append(result.score)

            final_scores = {}
            
            for report_type, scores in report_scores.items():
                max_score = max(scores)
                avg_score = np.mean(scores)
                final_scores[report_type] = 0.7 * max_score + 0.3 * avg_score
            
            best_report_type = max(final_scores.items(), key=lambda x: x[1])[0]
            
            logger.info(f"Classified query as: {best_report_type}")
            return best_report_type
            
        except Exception as e:
            logger.error(f"Error during classification: {e}")
            return "DetailedInvestigation"
    
    def get_classification_details(
        self,
        user_query: str,
        top_k: int = 10,
        score_threshold: float = 0.0
    ) -> Dict:
        """
        Get detailed classification results including scores and matched phrases.
        
        Args:
            user_query: The user's query text
            top_k: Number of top similar phrases to retrieve
            score_threshold: Minimum similarity score threshold
            
        Returns:
            Detailed classification results
        """
        try:
            query_embedding = self.embeddings_model.embed_query(user_query)
            
            search_results = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=top_k,
                score_threshold=score_threshold
            )
            if not search_results or not hasattr(search_results, 'points') or not search_results.points:
                return {"error": "No similar phrases found above threshold"}
            
            report_analysis = {}
            
            for result in search_results.points:
                report_type = result.payload["report_type"]
                phrase = result.payload["phrase"]
                score = result.score
                
                if report_type not in report_analysis:
                    report_analysis[report_type] = {
                        "scores": [],
                        "matched_phrases": [],
                        "max_score": 0,
                        "avg_score": 0,
                        "weighted_score": 0,
                        "best_phrase": ""
                    }
                
                report_analysis[report_type]["scores"].append(score)
                report_analysis[report_type]["matched_phrases"].append(phrase)
                
                if score > report_analysis[report_type]["max_score"]:
                    report_analysis[report_type]["max_score"] = score
                    report_analysis[report_type]["best_phrase"] = phrase
            
            for report_type, data in report_analysis.items():
                scores = data["scores"]
                data["avg_score"] = np.mean(scores)
                data["weighted_score"] = 0.7 * data["max_score"] + 0.3 * data["avg_score"]
            
            best_report = max(
                report_analysis.items(),
                key=lambda x: x[1]["weighted_score"]
            )[0]
            
            return {
                "best_classification": best_report,
                "query": user_query,
                "detailed_analysis": report_analysis,
                "total_matches": len(search_results.points)
            }
            
        except Exception as e:
            logger.error(f"Error getting classification details: {e}")
            return {"error": str(e)}
    
    def add_custom_report_type(
        self,
        report_type: str,
        phrases: List[str]
    ) -> bool:
        """
        Add a custom report type with its characteristic phrases.
        
        Args:
            report_type: Name of the new report type
            phrases: List of characteristic phrases
            
        Returns:
            True if successful, False otherwise
        """
        try:
            phrase_embeddings = self.embeddings_model.embed_documents(phrases)
            
            points = []
            for i, (phrase, embedding) in enumerate(zip(phrases, phrase_embeddings)):
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "report_type": report_type,
                        "phrase": phrase,
                        "phrase_index": i
                    }
                )
                points.append(point)
            
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            logger.info(f"Added {len(phrases)} phrases for report type: {report_type}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding custom report type: {e}")
            return False
    
    def get_available_report_types(self) -> List[str]:
        """Get list of all available report types."""
        try:
            scroll_result = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=1000
            )
            
            report_types = set()
            for point in scroll_result[0]:
                report_types.add(point.payload["report_type"])
            
            return sorted(list(report_types))
            
        except Exception as e:
            logger.error(f"Error getting report types: {e}")
            return []


import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Tuple, Dict
import logging

class ReportTypeClassifierEvaluator:
    def __init__(self, classifier, test_data: List[Tuple[str, str]], output_dir: str = "./extensions/dynamic-report-types/"):
        """
        Initialize the evaluator with a classifier and test dataset
        
        Args:
            classifier: Pre-initialized ReportTypeClassifier instance
            test_data: List of tuples (query, expected_report_type)
            output_dir: Directory to save outputs (default: ./extensions/dynamic-report-types/)
        """
        self.classifier = classifier
        self.test_data = test_data
        self.results = []
        self.metrics = {}
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def run_evaluation(self, detailed: bool = False) -> Dict:
        """
        Run classification on all test queries and calculate metrics
        
        Args:
            detailed: Whether to include per-query results in output
            
        Returns:
            Dictionary containing evaluation metrics and results
        """
        correct = 0
        confusion_matrix = {}
        class_counts = {}
        self.results = []
        
        # Initialize confusion matrix
        all_types = self.classifier.get_available_report_types()
        for true_type in all_types:
            confusion_matrix[true_type] = {pred_type: 0 for pred_type in all_types}
            class_counts[true_type] = 0

        # Process each test query
        for query, expected in self.test_data:
            actual = self.classifier.classify(query)
            is_correct = actual == expected
            
            # Update counts
            if is_correct:
                correct += 1
            class_counts[expected] += 1
            confusion_matrix[expected][actual] += 1
            
            # Store detailed results if requested
            if detailed:
                self.results.append({
                    "query": query,
                    "expected": expected,
                    "predicted": actual,
                    "correct": is_correct
                })

        # Calculate metrics
        total = len(self.test_data)
        accuracy = correct / total if total > 0 else 0
        
        # Calculate per-class precision and recall
        precision = {}
        recall = {}
        for report_type in all_types:
            true_positives = confusion_matrix[report_type].get(report_type, 0)
            total_predicted = sum(confusion_matrix[other].get(report_type, 0) for other in all_types)
            total_actual = class_counts[report_type]
            
            precision[report_type] = true_positives / total_predicted if total_predicted > 0 else 0
            recall[report_type] = true_positives / total_actual if total_actual > 0 else 0
        
        # Macro-average precision and recall
        macro_precision = sum(precision.values()) / len(all_types) if all_types else 0
        macro_recall = sum(recall.values()) / len(all_types) if all_types else 0
        
        # Store metrics
        self.metrics = {
            "accuracy": accuracy,
            "total_queries": total,
            "correct_predictions": correct,
            "precision": precision,
            "recall": recall,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "confusion_matrix": confusion_matrix,
            "class_counts": class_counts
        }
        
        return self.metrics
    
    def plot_confusion_matrix(self, save_path: Optional[str] = None) -> str:
        """
        Create and save confusion matrix plot
        
        Args:
            save_path: Custom path to save the plot (optional)
            
        Returns:
            Path where the plot was saved
        """
        if not self.metrics:
            raise ValueError("No evaluation results found. Run run_evaluation() first.")
        
        # Prepare data for plotting
        classes = sorted(self.metrics['confusion_matrix'].keys())
        matrix_data = []
        
        for true_class in classes:
            row = []
            for pred_class in classes:
                count = self.metrics['confusion_matrix'][true_class].get(pred_class, 0)
                row.append(count)
            matrix_data.append(row)
        
        # Create the plot
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            matrix_data,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=[cls.replace('_', '\n') for cls in classes],
            yticklabels=[cls.replace('_', '\n') for cls in classes],
            cbar_kws={'label': 'Count'}
        )
        
        plt.title(f'Confusion Matrix\nAccuracy: {self.metrics["accuracy"]:.2%}', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Class', fontsize=12, fontweight='bold')
        plt.ylabel('True Class', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save the plot
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.output_dir, f"confusion_matrix_{timestamp}.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def save_detailed_report(self, save_path: Optional[str] = None) -> str:
        """
        Save detailed evaluation report to a text file
        
        Args:
            save_path: Custom path to save the report (optional)
            
        Returns:
            Path where the report was saved
        """
        if not self.metrics:
            raise ValueError("No evaluation results found. Run run_evaluation() first.")
        
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.output_dir, f"evaluation_report_{timestamp}.txt")
        
        with open(save_path, 'w', encoding='utf-8') as f:
            metrics = self.metrics
            
            # Header
            f.write("="*70 + "\n")
            f.write("REPORT TYPE CLASSIFICATION EVALUATION REPORT\n")
            f.write("="*70 + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total test queries: {metrics['total_queries']}\n\n")
            
            # Overall metrics
            f.write("OVERALL PERFORMANCE METRICS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']:.2%})\n")
            f.write(f"Macro Precision: {metrics['macro_precision']:.4f} ({metrics['macro_precision']:.2%})\n")
            f.write(f"Macro Recall: {metrics['macro_recall']:.4f} ({metrics['macro_recall']:.2%})\n")
            f.write(f"Correct Predictions: {metrics['correct_predictions']}/{metrics['total_queries']}\n\n")
            
            # Class distribution
            f.write("CLASS DISTRIBUTION\n")
            f.write("-" * 18 + "\n")
            for cls, count in metrics['class_counts'].items():
                percentage = (count / metrics['total_queries']) * 100
                f.write(f"{cls:<30}: {count:>3} samples ({percentage:>5.1f}%)\n")
            f.write("\n")
            
            # Per-class metrics
            f.write("PER-CLASS PERFORMANCE METRICS\n")
            f.write("-" * 32 + "\n")
            f.write(f"{'Class':<30} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}\n")
            f.write("-" * 68 + "\n")
            
            classes = sorted(metrics['precision'].keys())
            for cls in classes:
                prec = metrics['precision'].get(cls, 0)
                rec = metrics['recall'].get(cls, 0)
                f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
                
                f.write(f"{cls:<30} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f}\n")
            f.write("\n")
            
            # Confusion matrix (text format)
            f.write("CONFUSION MATRIX\n")
            f.write("-" * 16 + "\n")
            f.write("Rows: True Class, Columns: Predicted Class\n\n")
            
            # Header
            header = " " * 20 + "".join([f"{c[:8]:<10}" for c in classes])
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")
            
            # Matrix rows
            for true_class in classes:
                row = f"{true_class[:18]:<20}"
                for pred_class in classes:
                    count = metrics['confusion_matrix'][true_class].get(pred_class, 0)
                    row += f"{count:<10}"
                f.write(row + "\n")
            f.write("\n")
            
            # Misclassifications
            if self.results:
                misclassifications = [r for r in self.results if not r['correct']]
                if misclassifications:
                    f.write("MISCLASSIFICATIONS\n")
                    f.write("-" * 18 + "\n")
                    for i, result in enumerate(misclassifications, 1):
                        f.write(f"{i}. Query: {result['query']}\n")
                        f.write(f"   Expected: {result['expected']}\n")
                        f.write(f"   Predicted: {result['predicted']}\n\n")
                else:
                    f.write("NO MISCLASSIFICATIONS - PERFECT ACCURACY!\n\n")
        
        return save_path
    
    def save_detailed_results_json(self, save_path: Optional[str] = None) -> str:
        """
        Save detailed results to JSON file for further analysis
        
        Args:
            save_path: Custom path to save the JSON file (optional)
            
        Returns:
            Path where the JSON file was saved
        """
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.output_dir, f"evaluation_results_{timestamp}.json")
        
        output_data = {
            "evaluation_metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_queries": len(self.test_data),
                "output_directory": self.output_dir
            },
            "metrics": self.metrics,
            "detailed_results": self.results if self.results else []
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        return save_path
    
    def generate_all_outputs(self, detailed: bool = True) -> Dict[str, str]:
        """
        Generate all outputs: confusion matrix plot, detailed report, and JSON results
        
        Args:
            detailed: Whether to include detailed results
            
        Returns:
            Dictionary with paths to all generated files
        """
        # Run evaluation if not already done
        if not self.metrics:
            self.run_evaluation(detailed=detailed)
        
        output_paths = {}
        
        # Generate confusion matrix plot
        try:
            plot_path = self.plot_confusion_matrix()
            output_paths['confusion_matrix_plot'] = plot_path
            print(f"✓ Confusion matrix plot saved to: {plot_path}")
        except Exception as e:
            print(f"✗ Error generating confusion matrix plot: {e}")
        
        # Generate detailed report
        try:
            report_path = self.save_detailed_report()
            output_paths['detailed_report'] = report_path
            print(f"✓ Detailed report saved to: {report_path}")
        except Exception as e:
            print(f"✗ Error generating detailed report: {e}")
        
        # Generate JSON results
        try:
            json_path = self.save_detailed_results_json()
            output_paths['json_results'] = json_path
            print(f"✓ JSON results saved to: {json_path}")
        except Exception as e:
            print(f"✗ Error generating JSON results: {e}")
        
        return output_paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    classifier = ReportTypeClassifier(
        qdrant_url="https://ft-vdb.epiphaiplatform.com:443",
        qdrant_api_key="Y3NMMGK3Okzt7rzho88jhmzPZl5Mhhnd98i39bLG4OJPGRtNV7pH8sOlVNtveGce",
        force_recreate=False,
    )
    
    # classifier._setup_collection()

    test_queries = [
        "Can you break down Apple's latest earnings report and revenue trends?",
        "What are the RSI and MACD trends for Bitcoin right now?",
        "What's the size and growth potential of the electric vehicle market?",
        "Tell me about the leadership team and structure at Netflix.",
        "Compare the features of the iPhone 16 vs. Samsung Galaxy S25."
    ]
    
    for query in test_queries:
        result = classifier.classify(query, top_k = 10)
        print(f"Query: {query}")
        print(f"Classification: {result}")
        print("-" * 50)
    
    detailed = classifier.get_classification_details(test_queries[0], top_k = 10)
    print("\nDetailed Analysis:")
    print(f"Best Classification: {detailed['best_classification']}")
    print(f"Total Matches: {detailed['total_matches']}")

    test_queries = [
        # FinancialAnalysis - Focus on financial metrics and statements
        ("Can you break down Apple's latest earnings report and revenue trends?", "FinancialAnalysis"),
        ("What's the current PE ratio and dividend yield for Tesla stock?", "FinancialAnalysis"),
        ("How does Amazon's cash flow and balance sheet look this year?", "FinancialAnalysis"),
        ("I need an analysis of Microsoft's profit margins and ROI.", "FinancialAnalysis"),
        ("Tell me about Google's financial ratios and how their assets are valued.", "FinancialAnalysis"),

        # TechnicalAnalysis - Focus on chart patterns and technical indicators
        ("What are the RSI and MACD trends for Bitcoin right now?", "TechnicalAnalysis"),
        ("Can you check the support and resistance levels for the S&P 500 ETF?", "TechnicalAnalysis"),
        ("Are there any breakout patterns in NVIDIA's stock chart?", "TechnicalAnalysis"),
        ("Show me the moving averages and Bollinger Bands for crude oil prices.", "TechnicalAnalysis"),
        ("I'm curious about candlestick patterns for gold futures—any insights?", "TechnicalAnalysis"),

        # MarketResearch - Focus on market dynamics and consumer behavior
        ("What's the size and growth potential of the electric vehicle market?", "MarketResearch"),
        ("Who are the main consumers in the smartphone industry right now?", "MarketResearch"),
        ("Can you analyze how people shop online for clothes?", "MarketResearch"),
        ("What's the demand outlook for solar energy in the next decade?", "MarketResearch"),
        ("How do streaming services like Netflix position their brands?", "MarketResearch"),

        # CompanyProfile - Focus on corporate structure and management
        ("Tell me about the leadership team and structure at Netflix.", "CompanyProfile"),
        ("What's the history of SpaceX and who runs it?", "CompanyProfile"),
        ("Can you describe Meta's business model and org chart?", "CompanyProfile"),
        ("What's Uber's mission and how is their governance set up?", "CompanyProfile"),
        ("Who's on Disney's board and what's their company culture like?", "CompanyProfile"),

        # IndustryAnalysis - Focus on sector dynamics and regulations
        ("What regulations shape the pharmaceutical industry today?", "IndustryAnalysis"),
        ("How is tech disrupting the financial services sector?", "IndustryAnalysis"),
        ("Can you explain the supply chain for computer chips?", "IndustryAnalysis"),
        ("What are the compliance rules for banks in the US?", "IndustryAnalysis"),
        ("How does the value chain work in the car manufacturing industry?", "IndustryAnalysis"),

        # ComparativeStudy - Focus on direct comparisons
        ("Compare the features of the iPhone 16 vs. Samsung Galaxy S25.", "ComparativeStudy"),
        ("What are the pros and cons of Tesla vs. Toyota cars?", "ComparativeStudy"),
        ("How does AWS stack up against Google Cloud in performance?", "ComparativeStudy"),
        ("Can you compare Netflix and Hulu's competitive strengths?", "ComparativeStudy"),
        ("Which is a better investment: Bitcoin or Ethereum?", "ComparativeStudy"),

        # TrendForecast - Focus on future predictions
        ("What will cybersecurity look like in the next five years?", "TrendForecast"),
        ("Can you predict how remote work will evolve by 2030?", "TrendForecast"),
        ("What's the future growth potential for cryptocurrencies?", "TrendForecast"),
        ("What trends are emerging in AI development?", "TrendForecast"),
        ("How will retail sales shift during the holiday season?", "TrendForecast"),

        # RiskAssessment - Focus on threats and vulnerabilities
        ("What are the biggest cybersecurity risks for banks?", "RiskAssessment"),
        ("Are there major risks to investing in Bitcoin right now?", "RiskAssessment"),
        ("What vulnerabilities exist in the manufacturing supply chain?", "RiskAssessment"),
        ("What operational risks come with space missions?", "RiskAssessment"),
        ("How risky is it to automate processes with AI?", "RiskAssessment"),

        # OpportunityMapping - Focus on growth and investment prospects
        ("Where are the best investment opportunities in renewable energy?", "OpportunityMapping"),
        ("What markets should a tech company expand into?", "OpportunityMapping"),
        ("Can you find partnership possibilities in medical AI?", "OpportunityMapping"),
        ("Is there untapped potential in blockchain for finance?", "OpportunityMapping"),
        ("What growth opportunities exist in virtual reality markets?", "OpportunityMapping"),

        # PerformanceMetrics - Focus on KPIs and measurement
        ("Can you build a dashboard to track my sales team's KPIs?", "PerformanceMetrics"),
        ("How can I measure my company's quarterly performance?", "PerformanceMetrics"),
        ("What metrics should I use to track team productivity?", "PerformanceMetrics"),
        ("Can you help monitor efficiency in our operations?", "PerformanceMetrics"),
        ("What are good success metrics for a digital ad campaign?", "PerformanceMetrics"),

        # StrategicPlanning - Focus on long-term planning
        ("Help me create a roadmap for moving my business online.", "StrategicPlanning"),
        ("What should my company's long-term goals be for growth?", "StrategicPlanning"),
        ("Can you design a plan to cut operational costs?", "StrategicPlanning"),
        ("What's a good strategic vision for my startup?", "StrategicPlanning"),
        ("How should we plan milestones for a new product launch?", "StrategicPlanning"),

        # ResourceCompilation - Focus on lists and directories
        ("Can you list some great resources for learning Python?", "ResourceCompilation"),
        ("What are the best tools for financial analysis?", "ResourceCompilation"),
        ("Compile a list of recent AI research papers.", "ResourceCompilation"),
        ("What are some good references for ESG investing?", "ResourceCompilation"),
        ("Can you gather cybersecurity training resources?", "ResourceCompilation"),

        # NewsDigest - Focus on current events and updates
        ("What's the latest news on Tesla's new projects?", "NewsDigest"),
        ("Give me a roundup of recent AI breakthroughs.", "NewsDigest"),
        ("What's happening with crypto regulations?", "NewsDigest"),
        ("Can you summarize media coverage of the iPhone 16 launch?", "NewsDigest"),
        ("What are the latest tech industry announcements?", "NewsDigest"),

        # HowToGuide - Focus on instructions and tutorials
        ("How do I start investing in the stock market?", "HowToGuide"),
        ("Can you give me a tutorial on building a machine learning model?", "HowToGuide"),
        ("Walk me through trading crypto as a beginner.", "HowToGuide"),
        ("How can I create a personal financial plan?", "HowToGuide"),
        ("What are the steps to launch a startup?", "HowToGuide"),

        # FAQCollection - Focus on questions and answers
        ("What are common questions about investing in Bitcoin?", "FAQCollection"),
        ("Can you answer typical questions about retirement savings?", "FAQCollection"),
        ("What are frequent issues with online trading platforms?", "FAQCollection"),
        ("What do people often ask about electric car benefits?", "FAQCollection"),
        ("Can you provide answers about filing taxes?", "FAQCollection"),

        # ScientificStudy - Focus on research methodology
        ("Can you study the effectiveness of new cancer treatments?", "ScientificStudy"),
        ("What does statistical analysis say about vaccine trials?", "ScientificStudy"),
        ("I need a peer-reviewed study on social media's mental health effects.", "ScientificStudy"),
        ("Can you design an experiment to test solar panel efficiency?", "ScientificStudy"),
        ("What's the latest research on dark matter hypotheses?", "ScientificStudy"),

        # ExecutiveBrief - Focus on concise summaries
        ("Give me a quick summary of climate change effects.", "ExecutiveBrief"),
        ("Can you provide a brief on our latest financial results?", "ExecutiveBrief"),
        ("What are the key points from our market research?", "ExecutiveBrief"),
        ("Summarize the impact of the recent merger.", "ExecutiveBrief"),
        ("Give me a high-level overview of our strategic plans.", "ExecutiveBrief"),

        # DetailedInvestigation - Focus on comprehensive analysis
        ("Can you do a deep dive into quantum computing uses?", "DetailedInvestigation"),
        ("What's an in-depth study on gene therapy advances?", "DetailedInvestigation"),
        ("I need a thorough analysis of climate change impacts.", "DetailedInvestigation"),
        ("Can you research blockchain scalability in detail?", "DetailedInvestigation"),
        ("Give me a comprehensive review of self-driving car tech.", "DetailedInvestigation")
    ]
    
    # Create and run evaluator
    evaluator = ReportTypeClassifierEvaluator(classifier, test_queries)
    
    # Generate all outputs
    output_files = evaluator.generate_all_outputs(detailed=False)
    
    print("\n" + "="*50)
    print("EVALUATION COMPLETE!")
    print("="*50)
    print(f"Overall Accuracy: {evaluator.metrics['accuracy']:.2%}")
    print(f"Total Test Queries: {evaluator.metrics['total_queries']}")
    print("\nGenerated Files:")
    for file_type, path in output_files.items():
        print(f"  - {file_type}: {path}")
    print("="*50)