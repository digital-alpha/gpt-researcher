from gpt_researcher.utils.enum import ReportType
from langchain_aws import BedrockEmbeddings
from typing import List, Dict, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class OptimizedReportTypeClassifier:
    def __init__(self):
        self.embeddings_model = BedrockEmbeddings(
            region_name="us-east-1", 
            model_id="amazon.titan-embed-text-v2:0"
        )
        self.report_data = []
        self.keyword_embeddings = {}
        self._setup_data()
    
    def _setup_data(self):
        self.report_data = [
            {
                "report_type": "FinancialAnalysis",
                "keywords": ["stock price", "quarterly earnings", "revenue growth", "profit margins", "balance sheet", "cash flow statement", "dividend yield", "PE ratio", "ROI", "ROE", "debt-to-equity", "market capitalization", "financial ratios", "income statement", "asset valuation"]
            },
            {
                "report_type": "TechnicalAnalysis", 
                "keywords": ["chart patterns", "candlestick analysis", "support resistance levels", "moving averages", "RSI indicator", "MACD signals", "Bollinger bands", "volume analysis", "breakout patterns", "trend lines", "fibonacci retracement", "momentum oscillators", "price action", "technical indicators"]
            },
            {
                "report_type": "MarketResearch",
                "keywords": ["market size", "industry growth rate", "consumer demographics", "market segmentation", "competitive landscape", "target audience analysis", "brand positioning", "customer surveys", "market penetration", "demand forecasting", "buying behavior", "market opportunities"]
            },
            {
                "report_type": "CompanyProfile",
                "keywords": ["corporate structure", "business model", "management biography", "company history", "organizational chart", "key executives", "board of directors", "corporate governance", "mission statement", "company culture", "headquarters location", "employee count"]
            },
            {
                "report_type": "IndustryAnalysis",
                "keywords": ["sector overview", "industry regulations", "supply chain dynamics", "technological disruption", "regulatory compliance", "industry standards", "market leaders", "emerging players", "industry challenges", "value chain analysis", "market dynamics"]
            },
            {
                "report_type": "ComparativeStudy",
                "keywords": ["side-by-side comparison", "pros and cons", "feature comparison", "benchmark analysis", "alternative options", "versus evaluation", "competitive advantages", "cost-benefit analysis", "performance comparison", "relative strengths", "ranking criteria"]
            },
            {
                "report_type": "TrendForecast",
                "keywords": ["future predictions", "emerging trends", "growth projections", "market outlook", "trend analysis", "forecasting models", "predictive analytics", "future scenarios", "trajectory analysis", "long-term trends", "cyclical patterns", "seasonal variations"]
            },
            {
                "report_type": "RiskAssessment",
                "keywords": ["threat analysis", "vulnerability assessment", "risk factors", "potential dangers", "mitigation strategies", "contingency planning", "risk matrix", "exposure analysis", "downside scenarios", "security risks", "operational risks", "financial risks"]
            },
            {
                "report_type": "OpportunityMapping",
                "keywords": ["growth opportunities", "investment prospects", "market gaps", "untapped potential", "expansion possibilities", "strategic opportunities", "emerging markets", "new revenue streams", "partnership opportunities", "innovation potential", "first-mover advantage"]
            },
            {
                "report_type": "PerformanceMetrics",
                "keywords": ["KPI dashboard", "performance indicators", "success metrics", "tracking results", "achievement measures", "benchmark performance", "scorecard analysis", "progress monitoring", "outcome evaluation", "efficiency metrics", "productivity measures"]
            },
            {
                "report_type": "StrategicPlanning",
                "keywords": ["strategic roadmap", "long-term objectives", "business strategy", "implementation plan", "strategic initiatives", "goal setting", "action plan", "strategic vision", "resource allocation", "milestone planning", "strategic priorities"]
            },
            {
                "report_type": "ResourceCompilation",
                "keywords": ["resource list", "tool directory", "reference materials", "helpful links", "source compilation", "database collection", "catalog of resources", "bibliography", "recommended reading", "useful websites", "toolkit collection"]
            },
            {
                "report_type": "NewsDigest",
                "keywords": ["latest news", "current events", "recent updates", "breaking developments", "news summary", "media coverage", "press releases", "announcement updates", "headline analysis", "news roundup", "current affairs"]
            },
            {
                "report_type": "HowToGuide",
                "keywords": ["step-by-step instructions", "tutorial guide", "implementation steps", "beginner tutorial", "learning path", "instructional guide", "practical walkthrough", "getting started", "how-to manual", "training materials", "educational content"]
            },
            {
                "report_type": "FAQCollection",
                "keywords": ["frequently asked questions", "common inquiries", "Q&A format", "troubleshooting guide", "help documentation", "support questions", "clarification points", "user queries", "problem solutions", "answer database"]
            },
            {
                "report_type": "ScientificStudy",
                "keywords": ["research methodology", "empirical data", "statistical analysis", "hypothesis testing", "experimental design", "peer review", "academic research", "scientific findings", "data collection", "literature review", "research conclusions"]
            },
            {
                "report_type": "ExecutiveBrief",
                "keywords": ["executive summary", "key highlights", "main points", "brief overview", "concise report", "summary findings", "quick briefing", "synopsis", "overview document", "condensed analysis", "high-level summary"]
            },
            {
                "report_type": "DetailedInvestigation",
                "keywords": ["comprehensive analysis", "thorough examination", "in-depth study", "extensive research", "detailed investigation", "exhaustive review", "complete assessment", "full analysis", "deep dive research", "meticulous study"]
            }
        ]
        self._precompute_embeddings()
    
    def _precompute_embeddings(self):
        print("Precomputing keyword embeddings...")
        for report_data in self.report_data:
            report_type = report_data["report_type"]
            keywords = report_data["keywords"]
            
            embeddings = self.embeddings_model.embed_documents(keywords)
            self.keyword_embeddings[report_type] = {
                "keywords": keywords,
                "embeddings": np.array(embeddings)
            }
        print("...Done")
    
    def find_best_report_type_v1(self, user_query: str) -> tuple[str, Dict]:
        """
        Calculate similarity with each keyword, then average per report type
        """
        query_embedding = np.array(self.embeddings_model.embed_query(user_query)).reshape(1, -1)
        
        report_scores = {}
        detailed_scores = {}
        
        for report_type, data in self.keyword_embeddings.items():
            keyword_embeddings = data["embeddings"]
            keywords = data["keywords"]
            
            similarities = cosine_similarity(query_embedding, keyword_embeddings)[0] # [[...]]
            
            detailed_scores[report_type] = {
                "keywords": keywords,
                "similarities": similarities.tolist(),
                "avg_similarity": np.mean(similarities)
            }
            
            report_scores[report_type] = np.mean(similarities)
        
        best_report_type = max(report_scores.items(), key=lambda x: x[1])[0]
        
        return best_report_type, detailed_scores
    
    def find_best_report_type_v2(self, user_query: str) -> tuple[str, Dict]:
        """
        Take the maximum similarity score for each report type
        """
        query_embedding = np.array(self.embeddings_model.embed_query(user_query)).reshape(1, -1)
        
        report_scores = {}
        detailed_scores = {}
        
        for report_type, data in self.keyword_embeddings.items():
            keyword_embeddings = data["embeddings"]
            keywords = data["keywords"]
            
            similarities = cosine_similarity(query_embedding, keyword_embeddings)[0]
            
            detailed_scores[report_type] = {
                "keywords": keywords,
                "similarities": similarities.tolist(),
                "max_similarity": float(np.max(similarities)),
                "best_keyword": keywords[np.argmax(similarities)]
            }
            
            report_scores[report_type] = np.max(similarities)
        
        best_report_type = max(report_scores.items(), key=lambda x: x[1])[0]
        
        return best_report_type, detailed_scores
    
    def find_best_report_type_v3(self, user_query: str, weight_factor: float = 0.7) -> tuple[str, Dict]:
        """
        Weighted avg of v1 and v2
        """
        query_embedding = np.array(self.embeddings_model.embed_query(user_query)).reshape(1, -1)
        
        report_scores = {}
        detailed_scores = {}
        
        for report_type, data in self.keyword_embeddings.items():
            keyword_embeddings = data["embeddings"]
            keywords = data["keywords"]
            
            similarities = cosine_similarity(query_embedding, keyword_embeddings)[0]
            
            max_sim = np.max(similarities)
            avg_sim = np.mean(similarities)
            
            combined_score = weight_factor * max_sim + (1 - weight_factor) * avg_sim
            
            detailed_scores[report_type] = {
                "keywords": keywords,
                "similarities": similarities.tolist(),
                "max_similarity": max_sim,
                "avg_similarity": avg_sim,
                "combined_score": combined_score,
                "best_keyword": keywords[np.argmax(similarities)]
            }
            
            report_scores[report_type] = combined_score
        
        best_report_type = max(report_scores.items(), key=lambda x: x[1])[0]
        
        return best_report_type, detailed_scores
    
    def find_best_report_type(self, user_query: str, method: str = "weighted") -> str:
        """
        method: average/max/weighted
        """
        if method == "average":
            best_type, _ = self.find_best_report_type_v1(user_query)
        elif method == "max":
            best_type, _ = self.find_best_report_type_v2(user_query)
        else:  # weighted
            best_type, _ = self.find_best_report_type_v3(user_query)
        
        return best_type
    
    def analyze_query(self, user_query: str) -> Dict:
        results = {}
        
        methods = [
            ("average", self.find_best_report_type_v1),
            ("max", self.find_best_report_type_v2),
            ("weighted", self.find_best_report_type_v3)
        ]
        
        for method_name, method_func in methods:
            best_type, detailed = method_func(user_query)
            results[method_name] = {
                "best_report_type": best_type,
                "detailed_scores": detailed
            }
        
        return results


# Usage example
if __name__ == "__main__":
    classifier = OptimizedReportTypeClassifier()
    
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
    
    print("=== OPTIMIZED REPORT TYPE CLASSIFICATION ===\n")
    
    sample_queries = test_queries[:5]  # First 5 for detailed output
    
    for query, expected_type in sample_queries:
        print(f"Query: {query}")
        print(f"Expected: {expected_type}")
        print("-" * 60)
        
        analysis = classifier.analyze_query(query)
        
        for method, result in analysis.items():
            predicted = result['best_report_type']
            is_correct = "✓" if predicted == expected_type else "✗"
            print(f"{method.upper()} method: {predicted} {is_correct}")
        
        print("\n" + "="*80 + "\n")
    
    # Test accuracy across all queries
    print("=== ACCURACY TESTING ===\n")
    methods = ["average", "max", "weighted"]
    accuracy_results = {method: 0 for method in methods}
    
    for query, expected_type in test_queries:
        for method in methods:
            predicted = classifier.find_best_report_type(query, method=method)
            if predicted == expected_type:
                accuracy_results[method] += 1
    
    total_queries = len(test_queries)
    print(f"Total test queries: {total_queries}")
    print("-" * 40)
    
    for method, correct_count in accuracy_results.items():
        accuracy = (correct_count / total_queries) * 100
        print(f"{method.upper()} method accuracy: {correct_count}/{total_queries} ({accuracy:.1f}%)")
    
    # Save misclassifications to a file
    print("\n=== MISCLASSIFICATION ANALYSIS ===\n")
    misclass_file = "./extensions/misclassifications.txt"
    with open(misclass_file, 'w') as f:
        f.write("=== MISCLASSIFICATION ANALYSIS ===\n\n")
        misclass_count = 0
        
        for query, expected_type in test_queries:  # Check all test queries
            predicted = classifier.find_best_report_type(query, method="weighted")
            if predicted != expected_type:
                misclass_count += 1
                misclass_text = f"Query: {query}\nExpected: {expected_type} | Predicted: {predicted}\n{'-' * 50}\n"
                f.write(misclass_text)
                print(misclass_text, end='')  # Also print to console
        
        f.write(f"\nTotal misclassifications: {misclass_count}/{total_queries} ({(misclass_count/total_queries)*100:.1f}%)")
        print(f"Total misclassifications: {misclass_count}/{total_queries} ({(misclass_count/total_queries)*100:.1f}%)")
        print(f"Misclassifications saved to: {misclass_file}")