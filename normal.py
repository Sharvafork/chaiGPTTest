import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import calendar
from collections import Counter
import os
import requests
import json
import sys
import subprocess
import importlib.util
import warnings
from typing import List, Dict, Any, Optional, Tuple, Union
import re
import io
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
from textwrap import wrap

# Suppress warnings
warnings.filterwarnings('ignore')

# Check and install required packages
required_packages = ["pandas", "numpy", "matplotlib", "seaborn", "pandasai", "reportlab", "fpdf", "requests"]
for package in required_packages:
    if importlib.util.find_spec(package) is None:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Now import the packages
from pandasai import Agent
from pandasai.connectors import PandasConnector
from pandasai.llm import Ollama
from fpdf import FPDF
import matplotlib.ticker as mtick

class DataAnalyzer:
    """
    A versatile data analyzer that can work with any dataset, 
    generate insights using Ollama LLMs, and create PDF reports.
    """
    
    def __init__(self, file_path: str, ollama_model: str = "mistral", ollama_url: str = "http://localhost:11434"):
        """
        Initialize the analyzer with the data file path and Ollama settings.
        
        Args:
            file_path (str): Path to the data file (CSV, Excel, etc.)
            ollama_model (str): Ollama model to use (default: "mistral")
            ollama_url (str): URL for the Ollama API (default: "http://localhost:11434")
        """
        self.file_path = file_path
        self.df = None
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url
        self.agent = None
        self.schema = {}
        self.column_types = {}
        self.insights = {}
        self.load_data()
        self.infer_schema()
        self.initialize_pandasai()
    
    def load_data(self) -> None:
        """Load data from various file formats."""
        try:
            file_ext = os.path.splitext(self.file_path)[1].lower()
            
            if file_ext == '.csv':
                self.df = pd.read_csv(self.file_path)
            elif file_ext in ['.xlsx', '.xls']:
                self.df = pd.read_excel(self.file_path)
            elif file_ext == '.json':
                self.df = pd.read_json(self.file_path)
            elif file_ext == '.parquet':
                self.df = pd.read_parquet(self.file_path)
            elif file_ext == '.sql':
                # This is a placeholder - would need a DB connection
                print("SQL files require database connection parameters")
                return
            else:
                # Try CSV as a fallback
                self.df = pd.read_csv(self.file_path)
            
            print(f"Data loaded successfully. Shape: {self.df.shape}")
            
            # Convert date columns to datetime
            for col in self.df.columns:
                # Check if column might contain dates
                if self.df[col].dtype == 'object':
                    try:
                        # Try to convert to datetime
                        pd.to_datetime(self.df[col], errors='raise')
                        self.df[col] = pd.to_datetime(self.df[col])
                        print(f"Converted {col} to datetime.")
                    except:
                        pass
        
        except Exception as e:
            print(f"Error loading data: {e}")
    
    def infer_schema(self) -> None:
        """Infer the schema and column types of the dataset."""
        if self.df is None:
            print("No data loaded.")
            return
        
        # Initialize schema dictionary
        self.schema = {
            "num_rows": len(self.df),
            "num_cols": len(self.df.columns),
            "column_names": list(self.df.columns),
            "column_types": {},
            "categorical_columns": [],
            "numerical_columns": [],
            "datetime_columns": [],
            "text_columns": [],
            "boolean_columns": [],
            "id_columns": [],
            "potential_target_columns": []
        }
        
        # Analyze each column
        for col in self.df.columns:
            # Get basic info
            dtype = self.df[col].dtype
            nunique = self.df[col].nunique()
            null_count = self.df[col].isna().sum()
            
            # Determine column type
            if pd.api.types.is_datetime64_dtype(self.df[col]):
                col_type = "datetime"
                self.schema["datetime_columns"].append(col)
            elif pd.api.types.is_numeric_dtype(self.df[col]):
                if nunique <= 2:
                    col_type = "boolean"
                    self.schema["boolean_columns"].append(col)
                else:
                    col_type = "numerical"
                    self.schema["numerical_columns"].append(col)
                    
                    # Check if it might be an ID column
                    if nunique == len(self.df) and null_count == 0:
                        self.schema["id_columns"].append(col)
            elif pd.api.types.is_bool_dtype(self.df[col]):
                col_type = "boolean"
                self.schema["boolean_columns"].append(col)
            elif nunique <= 20 and nunique / len(self.df) < 0.05:
                # If few unique values relative to dataset size, likely categorical
                col_type = "categorical"
                self.schema["categorical_columns"].append(col)
            else:
                # Check if it might be an ID or text column
                if any(id_pattern in col.lower() for id_pattern in ['id', 'code', 'key', 'num']):
                    self.schema["id_columns"].append(col)
                    col_type = "id"
                else:
                    col_type = "text"
                    self.schema["text_columns"].append(col)
            
            # Store column type
            self.schema["column_types"][col] = col_type
            self.column_types[col] = col_type
            
            # Identify potential target columns
            if col_type in ["categorical", "boolean"] and nunique <= 10:
                self.schema["potential_target_columns"].append(col)
        
        # Identify specific column types based on name patterns
        self.identify_special_columns()
        
        print(f"Schema inferred with {len(self.schema['numerical_columns'])} numerical, "
              f"{len(self.schema['categorical_columns'])} categorical, "
              f"{len(self.schema['datetime_columns'])} datetime columns.")
    
    def identify_special_columns(self) -> None:
        """Identify special columns based on name patterns."""
        # Define patterns for different column types
        patterns = {
            "date": ['date', 'time', 'year', 'month', 'day', 'created', 'updated', 'timestamp'],
            "amount": ['amount', 'price', 'cost', 'revenue', 'sales', 'income', 'expense', 'fee', 'payment'],
            "quantity": ['quantity', 'count', 'number', 'total', 'sum', 'qty'],
            "percentage": ['percent', 'rate', 'ratio', '%'],
            "location": ['country', 'city', 'state', 'address', 'location', 'region', 'postal', 'zip'],
            "person": ['name', 'customer', 'client', 'user', 'member', 'employee', 'staff'],
            "product": ['product', 'item', 'sku', 'service', 'subscription'],
            "category": ['category', 'type', 'group', 'segment', 'class', 'department'],
            "status": ['status', 'state', 'condition', 'flag'],
            "id": ['id', 'key', 'code', 'number', 'identifier', 'reference']
        }
        
        # Initialize special columns dictionary
        self.schema["special_columns"] = {category: [] for category in patterns.keys()}
        
        # Check each column against patterns
        for col in self.df.columns:
            col_lower = col.lower()
            for category, keywords in patterns.items():
                if any(keyword in col_lower for keyword in keywords):
                    self.schema["special_columns"][category].append(col)
                    break
    
    def initialize_pandasai(self) -> None:
        """Initialize PandasAI with Ollama LLM."""
        try:
            # Check if Ollama is available
            if self.test_ollama_connection():
                # Initialize Ollama LLM
                llm = Ollama(model=self.ollama_model, url=self.ollama_url)
                
                # Create a PandasAI agent
                self.agent = Agent(
                    [PandasConnector(self.df, name="data")],
                    llm=llm,
                    memory_size=10  # Remember last 10 conversations
                )
                
                print(f"PandasAI initialized successfully with Ollama model: {self.ollama_model}")
            else:
                print("Ollama server is not available. Make sure Ollama is running.")
                print("Install Ollama from https://ollama.ai/ and start it before running this script.")
                print("Continuing with standard analysis methods.")
        except Exception as e:
            print(f"Error initializing PandasAI with Ollama: {e}")
            print("Continuing with standard analysis methods.")
    
    def test_ollama_connection(self) -> bool:
        """Test the connection to Ollama server."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [model.get("name") for model in models]
                print(f"Ollama connection successful. Available models: {', '.join(model_names)}")
                
                if self.ollama_model not in model_names:
                    print(f"Warning: Model '{self.ollama_model}' not found in available models.")
                    print(f"You may need to pull it first with: 'ollama pull {self.ollama_model}'")
                
                return True
            else:
                print(f"Ollama server responded with status code {response.status_code}")
                return False
        except Exception as e:
            print(f"Error connecting to Ollama server: {e}")
            print("Make sure Ollama is installed and running.")
            print("Installation instructions: https://ollama.ai/")
            return False
    
    def ask(self, query: str) -> str:
        """
        Use PandasAI to answer natural language queries about the data.
        
        Args:
            query (str): The natural language query to ask about the data
            
        Returns:
            str: The response to the query
        """
        if self.df is None:
            return "No data loaded."
        
        if not self.agent:
            return "PandasAI not initialized. Make sure Ollama is running."
        
        try:
            print(f"Asking {self.ollama_model}: {query}")
            
            # Try using the agent directly
            try:
                result = self.agent.chat(query)
                return result
            except Exception as e:
                print(f"Agent chat error: {e}")
                
                # Fall back to direct analysis
                result = self.direct_analysis(query)
                if result:
                    return result
                
                return f"I couldn't process that query with the AI agent. Here's what I can tell you based on the data:\n\n{self.get_data_summary()}"
                
        except Exception as e:
            print(f"Error processing query: {e}")
            return f"Error processing query. Here's a summary of the data instead:\n\n{self.get_data_summary()}"
    
    def direct_analysis(self, query: str) -> Optional[str]:
        """
        Perform direct analysis based on the query without using the LLM.
        
        Args:
            query (str): The query to analyze
            
        Returns:
            Optional[str]: The analysis result or None if no direct analysis is possible
        """
        query_lower = query.lower()
        
        # Extract numbers from the query
        numbers = re.findall(r'\d+', query)
        n = int(numbers[0]) if numbers else 5  # Default to top 5 if no number specified
        
        # Check for different query types
        if any(word in query_lower for word in ['top', 'highest', 'most']):
            # Handle top N queries
            if any(word in query_lower for word in ['correlation', 'correlate', 'relationship']):
                return self.analyze_correlations(n)
            
            for col_type, columns in self.schema["special_columns"].items():
                if any(col.lower() in query_lower for col in columns) or col_type.lower() in query_lower:
                    for col in columns:
                        if col in self.schema["numerical_columns"]:
                            return self.analyze_top_values(col, n)
            
            # Check for categorical columns
            for col in self.schema["categorical_columns"]:
                if col.lower() in query_lower:
                    return self.analyze_category_distribution(col, n)
        
        elif any(word in query_lower for word in ['distribution', 'histogram', 'frequency']):
            # Handle distribution queries
            for col_type, columns in self.schema["special_columns"].items():
                if any(col.lower() in query_lower for col in columns) or col_type.lower() in query_lower:
                    for col in columns:
                        if col in self.schema["numerical_columns"]:
                            return self.analyze_numerical_distribution(col)
                        elif col in self.schema["categorical_columns"]:
                            return self.analyze_category_distribution(col)
        
        elif any(word in query_lower for word in ['trend', 'time', 'over time', 'by date']):
            # Handle time trend queries
            for date_col in self.schema["datetime_columns"]:
                for num_col in self.schema["numerical_columns"]:
                    if num_col.lower() in query_lower or any(word in num_col.lower() for word in ['amount', 'price', 'sales', 'revenue']):
                        return self.analyze_time_trend(date_col, num_col)
        
        elif any(word in query_lower for word in ['summary', 'overview', 'statistics']):
            # Handle summary queries
            return self.get_data_summary()
        
        # If no direct analysis matched, return None
        return None
    
    def analyze_top_values(self, column: str, n: int = 5) -> str:
        """
        Analyze the top N values in a numerical column.
        
        Args:
            column (str): The column to analyze
            n (int): Number of top values to return
            
        Returns:
            str: Analysis of top values
        """
        if column not in self.df.columns:
            return f"Column '{column}' not found in the dataset."
        
        try:
            # Sort by the column in descending order
            top_values = self.df.sort_values(by=column, ascending=False).head(n)
            
            result = f"Top {n} values for {column}:\n\n"
            
            # If there are ID or categorical columns, include them in the result
            id_cols = [col for col in self.schema["id_columns"] if col != column]
            cat_cols = [col for col in self.schema["categorical_columns"] if col != column]
            
            display_cols = []
            if id_cols:
                display_cols.append(id_cols[0])  # Add one ID column
            if cat_cols:
                display_cols.append(cat_cols[0])  # Add one categorical column
            
            display_cols.append(column)  # Add the target column
            
            # Create a formatted table
            for i, row in enumerate(top_values[display_cols].itertuples(), 1):
                values = [f"{getattr(row, col)}" for col in display_cols]
                result += f"{i}. {' - '.join(values)}\n"
            
            # Add some statistics
            total = self.df[column].sum()
            top_total = top_values[column].sum()
            percentage = (top_total / total * 100) if total else 0
            
            result += f"\nThese top {n} represent {percentage:.2f}% of the total {column} ({top_total:.2f} out of {total:.2f})"
            
            return result
        
        except Exception as e:
            return f"Error analyzing top values for {column}: {e}"
    
    def analyze_category_distribution(self, column: str, n: int = None) -> str:
        """
        Analyze the distribution of values in a categorical column.
        
        Args:
            column (str): The column to analyze
            n (int): Number of top categories to return (optional)
            
        Returns:
            str: Analysis of category distribution
        """
        if column not in self.df.columns:
            return f"Column '{column}' not found in the dataset."
        
        try:
            # Get value counts
            value_counts = self.df[column].value_counts()
            
            # Limit to top N if specified
            if n and n < len(value_counts):
                value_counts = value_counts.head(n)
                result = f"Top {n} categories for {column}:\n\n"
            else:
                result = f"Distribution of {column}:\n\n"
            
            # Calculate percentages
            total = value_counts.sum()
            
            # Create a formatted table
            for category, count in value_counts.items():
                percentage = (count / total * 100)
                result += f"{category}: {count} ({percentage:.2f}%)\n"
            
            return result
        
        except Exception as e:
            return f"Error analyzing category distribution for {column}: {e}"
    
    def analyze_numerical_distribution(self, column: str) -> str:
        """
        Analyze the distribution of values in a numerical column.
        
        Args:
            column (str): The column to analyze
            
        Returns:
            str: Analysis of numerical distribution
        """
        if column not in self.df.columns:
            return f"Column '{column}' not found in the dataset."
        
        try:
            # Calculate statistics
            stats = self.df[column].describe()
            
            result = f"Distribution of {column}:\n\n"
            result += f"Count: {stats['count']:.0f}\n"
            result += f"Mean: {stats['mean']:.2f}\n"
            result += f"Median: {self.df[column].median():.2f}\n"
            result += f"Std Dev: {stats['std']:.2f}\n"
            result += f"Min: {stats['min']:.2f}\n"
            result += f"Max: {stats['max']:.2f}\n\n"
            
            # Calculate quartiles
            result += f"25th Percentile: {stats['25%']:.2f}\n"
            result += f"50th Percentile: {stats['50%']:.2f}\n"
            result += f"75th Percentile: {stats['75%']:.2f}\n\n"
            
            # Calculate skewness and kurtosis if scipy is available
            try:
                from scipy import stats as scipy_stats
                skewness = scipy_stats.skew(self.df[column].dropna())
                kurtosis = scipy_stats.kurtosis(self.df[column].dropna())
                
                result += f"Skewness: {skewness:.2f} "
                if skewness > 0.5:
                    result += "(Right-skewed/Positively skewed)\n"
                elif skewness < -0.5:
                    result += "(Left-skewed/Negatively skewed)\n"
                else:
                    result += "(Approximately symmetric)\n"
                
                result += f"Kurtosis: {kurtosis:.2f} "
                if kurtosis > 0.5:
                    result += "(Heavy-tailed/More outliers than normal distribution)\n"
                elif kurtosis < -0.5:
                    result += "(Light-tailed/Fewer outliers than normal distribution)\n"
                else:
                    result += "(Similar to normal distribution)\n"
            except:
                pass
            
            return result
        
        except Exception as e:
            return f"Error analyzing numerical distribution for {column}: {e}"
    
    def analyze_correlations(self, n: int = 5) -> str:
        """
        Analyze correlations between numerical columns.
        
        Args:
            n (int): Number of top correlations to return
            
        Returns:
            str: Analysis of correlations
        """
        if len(self.schema["numerical_columns"]) < 2:
            return "Not enough numerical columns to calculate correlations."
        
        try:
            # Calculate correlation matrix
            corr_matrix = self.df[self.schema["numerical_columns"]].corr()
            
            # Get the upper triangle of the correlation matrix
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            
            # Find the top N correlations
            top_corr = upper.unstack().sort_values(kind="quicksort", ascending=False).head(n)
            
            result = f"Top {n} correlations between numerical columns:\n\n"
            
            # Create a formatted table
            for (col1, col2), corr in top_corr.items():
                result += f"{col1} and {col2}: {corr:.4f}\n"
                
                # Add interpretation
                if corr > 0.7:
                    result += "  (Strong positive correlation)\n"
                elif corr > 0.3:
                    result += "  (Moderate positive correlation)\n"
                elif corr > 0:
                    result += "  (Weak positive correlation)\n"
                elif corr > -0.3:
                    result += "  (Weak negative correlation)\n"
                elif corr > -0.7:
                    result += "  (Moderate negative correlation)\n"
                else:
                    result += "  (Strong negative correlation)\n"
            
            return result
        
        except Exception as e:
            return f"Error analyzing correlations: {e}"
    
    def analyze_time_trend(self, date_column: str, value_column: str) -> str:
        """
        Analyze trends over time.
        
        Args:
            date_column (str): The date column to use
            value_column (str): The value column to analyze
            
        Returns:
            str: Analysis of time trends
        """
        if date_column not in self.df.columns or value_column not in self.df.columns:
            return f"Columns '{date_column}' or '{value_column}' not found in the dataset."
        
        try:
            # Ensure date column is datetime
            if not pd.api.types.is_datetime64_dtype(self.df[date_column]):
                try:
                    self.df[date_column] = pd.to_datetime(self.df[date_column])
                except:
                    return f"Column '{date_column}' could not be converted to datetime."
            
            # Group by month and calculate statistics
            self.df['year_month'] = self.df[date_column].dt.to_period('M')
            monthly_stats = self.df.groupby('year_month')[value_column].agg(['mean', 'sum', 'count'])
            
            # Format the result
            result = f"Monthly trend of {value_column}:\n\n"
            result += "Month | Count | Sum | Average\n"
            result += "----- | ----- | --- | -------\n"
            
            for period, stats in monthly_stats.iterrows():
                result += f"{period} | {stats['count']:.0f} | {stats['sum']:.2f} | {stats['mean']:.2f}\n"
            
            # Calculate growth rates
            if len(monthly_stats) > 1:
                first_sum = monthly_stats['sum'].iloc[0]
                last_sum = monthly_stats['sum'].iloc[-1]
                total_growth = ((last_sum / first_sum) - 1) * 100 if first_sum else 0
                
                result += f"\nTotal growth from {monthly_stats.index[0]} to {monthly_stats.index[-1]}: {total_growth:.2f}%\n"
                
                # Calculate average monthly growth rate
                monthly_growth_rates = monthly_stats['sum'].pct_change() * 100
                avg_monthly_growth = monthly_growth_rates.mean()
                
                result += f"Average monthly growth rate: {avg_monthly_growth:.2f}%\n"
            
            return result
        
        except Exception as e:
            return f"Error analyzing time trend: {e}"
    
    def get_data_summary(self) -> str:
        """
        Generate a basic summary of the data.
        
        Returns:
            str: Summary of the data
        """
        if self.df is None:
            return "No data loaded."
        
        summary = "Data Summary:\n\n"
        summary += f"Dataset Shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns\n\n"
        
        # Column types summary
        summary += "Column Types:\n"
        summary += f"- Numerical: {len(self.schema['numerical_columns'])}\n"
        summary += f"- Categorical: {len(self.schema['categorical_columns'])}\n"
        summary += f"- Datetime: {len(self.schema['datetime_columns'])}\n"
        summary += f"- Text: {len(self.schema['text_columns'])}\n"
        summary += f"- Boolean: {len(self.schema['boolean_columns'])}\n\n"
        
        # Missing values
        missing_values = self.df.isna().sum()
        if missing_values.sum() > 0:
            summary += "Columns with Missing Values:\n"
            for col, count in missing_values[missing_values > 0].items():
                percentage = (count / len(self.df)) * 100
                summary += f"- {col}: {count} ({percentage:.2f}%)\n"
            summary += "\n"
        else:
            summary += "No missing values found in the dataset.\n\n"
        
        # Sample data
        summary += "Sample Data (first 5 rows):\n"
        sample_data = self.df.head().to_string()
        summary += sample_data + "\n\n"
        
        # Basic statistics for numerical columns
        if self.schema['numerical_columns']:
            summary += "Basic Statistics for Numerical Columns:\n"
            stats = self.df[self.schema['numerical_columns']].describe().to_string()
            summary += stats + "\n"
        
        return summary
    
    def generate_insights(self) -> Dict[str, str]:
        """
        Generate insights about the dataset.
        
        Returns:
            Dict[str, str]: Dictionary of insights
        """
        print("\n=== GENERATING INSIGHTS ===")
        
        insights = {}
        
        # Basic dataset insights
        insights["dataset_overview"] = self.get_data_summary()
        
        # Analyze numerical columns
        if self.schema['numerical_columns']:
            print("Analyzing numerical columns...")
            for col in self.schema['numerical_columns'][:5]:  # Limit to first 5 to avoid too many insights
                insights[f"numerical_{col}"] = self.analyze_numerical_distribution(col)
        
        # Analyze categorical columns
        if self.schema['categorical_columns']:
            print("Analyzing categorical columns...")
            for col in self.schema['categorical_columns'][:5]:  # Limit to first 5
                insights[f"categorical_{col}"] = self.analyze_category_distribution(col)
        
        # Analyze correlations
        if len(self.schema['numerical_columns']) >= 2:
            print("Analyzing correlations...")
            insights["correlations"] = self.analyze_correlations(10)
        
        # Analyze time trends if datetime columns exist
        if self.schema['datetime_columns'] and self.schema['numerical_columns']:
            print("Analyzing time trends...")
            date_col = self.schema['datetime_columns'][0]
            for value_col in self.schema['numerical_columns'][:3]:  # Limit to first 3
                insights[f"time_trend_{date_col}_{value_col}"] = self.analyze_time_trend(date_col, value_col)
        
        # Use AI for additional insights if available
        if self.agent:
            print("Generating AI insights...")
            try:
                ai_queries = [
                    "What are the most interesting patterns in this dataset?",
                    "What are the key factors that influence the main metrics in this dataset?",
                    "What recommendations would you make based on this data?",
                    "Are there any anomalies or outliers in this dataset that should be investigated?",
                    "What are the main trends over time in this dataset?"
                ]
                
                for i, query in enumerate(ai_queries):
                    print(f"AI Query {i+1}: {query}")
                    insights[f"ai_insight_{i+1}"] = self.ask(query)
            except Exception as e:
                print(f"Error generating AI insights: {e}")
        
        self.insights = insights
        return insights
    
    def create_visualizations(self) -> List[Figure]:
        """
        Create visualizations for the dataset.
        
        Returns:
            List[Figure]: List of matplotlib figures
        """
        print("\n=== CREATING VISUALIZATIONS ===")
        
        figures = []
        
        # Set the style
        plt.style.use('ggplot')
        sns.set_palette("viridis")
        
        # 1. Missing values heatmap
        if self.df.isna().sum().sum() > 0:
            print("Creating missing values heatmap...")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(self.df.isna(), cbar=False, cmap='viridis', ax=ax)
            ax.set_title('Missing Values Heatmap')
            ax.set_xlabel('Columns')
            ax.set_ylabel('Rows')
            figures.append(fig)
        
        # 2. Numerical distributions
        if self.schema['numerical_columns']:
            print("Creating numerical distributions...")
            for i in range(0, min(len(self.schema['numerical_columns']), 6), 2):
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                
                for j in range(2):
                    if i + j < len(self.schema['numerical_columns']):
                        col = self.schema['numerical_columns'][i + j]
                        sns.histplot(self.df[col].dropna(), kde=True, ax=axes[j])
                        axes[j].set_title(f'Distribution of {col}')
                        axes[j].set_xlabel(col)
                        axes[j].set_ylabel('Frequency')
                
                plt.tight_layout()
                figures.append(fig)
        
        # 3. Categorical distributions
        if self.schema['categorical_columns']:
            print("Creating categorical distributions...")
            for i in range(0, min(len(self.schema['categorical_columns']), 6), 2):
                fig, axes = plt.subplots(1, 2, figsize=(14, 6))
                
                for j in range(2):
                    if i + j < len(self.schema['categorical_columns']):
                        col = self.schema['categorical_columns'][i + j]
                        value_counts = self.df[col].value_counts().head(10)  # Top 10 categories
                        sns.barplot(x=value_counts.index, y=value_counts.values, ax=axes[j])
                        axes[j].set_title(f'Top Categories in {col}')
                        axes[j].set_xlabel(col)
                        axes[j].set_ylabel('Count')
                        axes[j].tick_params(axis='x', rotation=45)
                
                plt.tight_layout()
                figures.append(fig)
        
        # 4. Correlation heatmap
        if len(self.schema['numerical_columns']) >= 2:
            print("Creating correlation heatmap...")
            fig, ax = plt.subplots(figsize=(10, 8))
            corr_matrix = self.df[self.schema['numerical_columns']].corr()
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', annot=True, fmt='.2f', ax=ax)
            ax.set_title('Correlation Heatmap')
            figures.append(fig)
        
        # 5. Time series plots
        if self.schema['datetime_columns'] and self.schema['numerical_columns']:
            print("Creating time series plots...")
            date_col = self.schema['datetime_columns'][0]
            
            # Ensure date column is datetime
            if not pd.api.types.is_datetime64_dtype(self.df[date_col]):
                try:
                    self.df[date_col] = pd.to_datetime(self.df[date_col])
                except:
                    print(f"Column '{date_col}' could not be converted to datetime.")
            
            # Create time series for top 3 numerical columns
            for i, col in enumerate(self.schema['numerical_columns'][:3]):
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Group by month and calculate mean
                self.df['year_month'] = self.df[date_col].dt.to_period('M')
                monthly_data = self.df.groupby('year_month')[col].mean()
                
                # Convert period index to datetime for plotting
                monthly_data.index = monthly_data.index.to_timestamp()
                
                # Plot
                ax.plot(monthly_data.index, monthly_data.values, marker='o', linestyle='-')
                ax.set_title(f'Monthly Average of {col}')
                ax.set_xlabel('Date')
                ax.set_ylabel(col)
                ax.grid(True)
                
                # Format x-axis
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                figures.append(fig)
        
        # 6. Boxplots for numerical columns by categorical
        if self.schema['numerical_columns'] and self.schema['categorical_columns']:
            print("Creating boxplots...")
            num_col = self.schema['numerical_columns'][0]  # First numerical column
            cat_col = self.schema['categorical_columns'][0]  # First categorical column
            
            # Limit categories to top 10
            top_categories = self.df[cat_col].value_counts().head(10).index
            filtered_df = self.df[self.df[cat_col].isin(top_categories)]
            
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.boxplot(x=cat_col, y=num_col, data=filtered_df, ax=ax)
            ax.set_title(f'Distribution of {num_col} by {cat_col}')
            ax.set_xlabel(cat_col)
            ax.set_ylabel(num_col)
            ax.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            
            figures.append(fig)
        
        return figures
    
    def generate_pdf_report(self, output_path: str = "data_analysis_report.pdf") -> str:
        """
        Generate a comprehensive PDF report with insights and visualizations.
        
        Args:
            output_path (str): Path to save the PDF report
            
        Returns:
            str: Path to the generated PDF report
        """
        print(f"\n=== GENERATING PDF REPORT: {output_path} ===")
        
        # Generate insights if not already done
        if not self.insights:
            self.generate_insights()
        
        # Create visualizations
        figures = self.create_visualizations()
        
        # Create PDF
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        
        # Add title
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Data Analysis Report", ln=True, align="C")
        pdf.ln(5)
        
        # Add dataset info
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Dataset Overview", ln=True)
        pdf.set_font("Arial", "", 10)
        
        # Dataset shape
        pdf.cell(0, 10, f"Dataset Shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns", ln=True)
        
        # Column types
        pdf.cell(0, 10, "Column Types:", ln=True)
        pdf.cell(10, 5, "", ln=False)
        pdf.cell(0, 5, f"Numerical: {len(self.schema['numerical_columns'])}", ln=True)
        pdf.cell(10, 5, "", ln=False)
        pdf.cell(0, 5, f"Categorical: {len(self.schema['categorical_columns'])}", ln=True)
        pdf.cell(10, 5, "", ln=False)
        pdf.cell(0, 5, f"Datetime: {len(self.schema['datetime_columns'])}", ln=True)
        pdf.cell(10, 5, "", ln=False)
        pdf.cell(0, 5, f"Text: {len(self.schema['text_columns'])}", ln=True)
        pdf.cell(10, 5, "", ln=False)
        pdf.cell(0, 5, f"Boolean: {len(self.schema['boolean_columns'])}", ln=True)
        pdf.ln(5)
        
        # Missing values
        missing_values = self.df.isna().sum()
        if missing_values.sum() > 0:
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, "Missing Values", ln=True)
            pdf.set_font("Arial", "", 10)
            
            for col, count in missing_values[missing_values > 0].items():
                percentage = (count / len(self.df)) * 100
                pdf.cell(0, 5, f"{col}: {count} ({percentage:.2f}%)", ln=True)
            
            pdf.ln(5)
        
        # Add insights
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Key Insights", ln=True)
        pdf.set_font("Arial", "", 10)
        
        # Add AI insights if available
        ai_insights = [insight for key, insight in self.insights.items() if key.startswith('ai_insight')]
        if ai_insights:
            for i, insight in enumerate(ai_insights):
                # Split long text into paragraphs
                paragraphs = insight.split('\n')
                for para in paragraphs:
                    # Wrap long paragraphs
                    wrapped_text = "\n".join(wrap(para, width=80))
                    for line in wrapped_text.split('\n'):
                        pdf.multi_cell(0, 5, line)
                pdf.ln(5)
        
        # Add numerical insights
        if self.schema['numerical_columns']:
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, "Numerical Column Insights", ln=True)
            pdf.set_font("Arial", "", 10)
            
            for col in self.schema['numerical_columns'][:3]:  # Limit to first 3
                insight_key = f"numerical_{col}"
                if insight_key in self.insights:
                    pdf.set_font("Arial", "B", 10)
                    pdf.cell(0, 5, col, ln=True)
                    pdf.set_font("Arial", "", 10)
                    
                    # Split and format the insight text
                    insight_text = self.insights[insight_key]
                    lines = insight_text.split('\n')
                    for line in lines:
                        if line.strip():
                            pdf.cell(0, 5, line, ln=True)
                    
                    pdf.ln(5)
        
        # Add categorical insights
        if self.schema['categorical_columns']:
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, "Categorical Column Insights", ln=True)
            pdf.set_font("Arial", "", 10)
            
            for col in self.schema['categorical_columns'][:3]:  # Limit to first 3
                insight_key = f"categorical_{col}"
                if insight_key in self.insights:
                    pdf.set_font("Arial", "B", 10)
                    pdf.cell(0, 5, col, ln=True)
                    pdf.set_font("Arial", "", 10)
                    
                    # Split and format the insight text
                    insight_text = self.insights[insight_key]
                    lines = insight_text.split('\n')
                    for line in lines:
                        if line.strip():
                            pdf.cell(0, 5, line, ln=True)
                    
                    pdf.ln(5)
        
        # Add correlation insights
        if "correlations" in self.insights:
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, "Correlation Insights", ln=True)
            pdf.set_font("Arial", "", 10)
            
            # Split and format the correlation text
            correlation_text = self.insights["correlations"]
            lines = correlation_text.split('\n')
            for line in lines:
                if line.strip():
                    pdf.cell(0, 5, line, ln=True)
            
            pdf.ln(5)
        
        # Add time trend insights
        time_trend_keys = [key for key in self.insights.keys() if key.startswith('time_trend')]
        if time_trend_keys:
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, "Time Trend Insights", ln=True)
            pdf.set_font("Arial", "", 10)
            
            for key in time_trend_keys[:2]:  # Limit to first 2
                # Split and format the time trend text
                time_trend_text = self.insights[key]
                lines = time_trend_text.split('\n')
                for line in lines:
                    if line.strip():
                        pdf.cell(0, 5, line, ln=True)
                
                pdf.ln(5)
        
        # Add visualizations
        if figures:
            pdf.add_page()
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, "Visualizations", ln=True)
            
            # Save each figure to a temporary file and add to PDF
            for i, fig in enumerate(figures):
                # Save figure to a temporary file
                temp_file = f"temp_figure_{i}.png"
                fig.savefig(temp_file, dpi=300, bbox_inches='tight')
                plt.close(fig)
                
                # Add figure to PDF
                pdf.image(temp_file, x=10, w=190)
                pdf.ln(5)
                
                # Remove temporary file
                os.remove(temp_file)
        
        # Save PDF
        pdf.output(output_path)
        print(f"PDF report saved to {output_path}")
        
        return output_path
    
    def analyze_dataset(self, output_pdf: str = "data_analysis_report.pdf") -> Dict[str, Any]:
        """
        Perform a comprehensive analysis of the dataset and generate a PDF report.
        
        Args:
            output_pdf (str): Path to save the PDF report
            
        Returns:
            Dict[str, Any]: Dictionary with analysis results
        """
        print("\n=== STARTING COMPREHENSIVE DATASET ANALYSIS ===")
        
        # Generate insights
        insights = self.generate_insights()
        
        # Generate PDF report
        pdf_path = self.generate_pdf_report(output_pdf)
        
        # Return analysis results
        return {
            "insights": insights,
            "schema": self.schema,
            "pdf_report": pdf_path
        }

# Example usage
if __name__ == "__main__":
    # Check if a file path was provided as a command line argument
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # Default to a sample dataset if no file path is provided
        file_path = "sample_data.csv"
        
        # Create a sample dataset if it doesn't exist
        if not os.path.exists(file_path):
            print("Creating a sample dataset for demonstration...")
            
            # Create sample data
            import random
            from datetime import datetime, timedelta
            
            # Define sample parameters
            num_records = 1000
            categories = ['Category A', 'Category B', 'Category C', 'Category D', 'Category E']
            regions = ['North', 'South', 'East', 'West', 'Central']
            product_types = ['Type 1', 'Type 2', 'Type 3', 'Type 4']
            
            # Generate random dates
            start_date = datetime(2020, 1, 1)
            end_date = datetime(2023, 12, 31)
            date_range = (end_date - start_date).days
            
            # Create sample data
            data = []
            for i in range(1, num_records + 1):
                transaction_id = f"TRX{i:06d}"
                category = random.choice(categories)
                region = random.choice(regions)
                product_type = random.choice(product_types)
                
                transaction_date = start_date + timedelta(days=random.randint(0, date_range))
                
                amount = random.uniform(50, 500)
                quantity = random.randint(1, 10)
                unit_price = amount / quantity
                
                discount = random.uniform(0, 0.3) if random.random() > 0.7 else 0
                
                data.append({
                    'TransactionID': transaction_id,
                    'Date': transaction_date,
                    'Category': category,
                    'Region': region,
                    'ProductType': product_type,
                    'Amount': amount,
                    'Quantity': quantity,
                    'UnitPrice': unit_price,
                    'Discount': discount,
                    'NetAmount': amount * (1 - discount)
                })
            
            # Create DataFrame and save to CSV
            sample_df = pd.DataFrame(data)
            sample_df.to_csv(file_path, index=False)
            print(f"Sample dataset created and saved to {file_path}")
    
    # Create analyzer with Mistral model
    analyzer = DataAnalyzer(file_path, ollama_model="mistral")
    
    # Test Ollama connection
    ollama_available = analyzer.test_ollama_connection()
    
    if not ollama_available:
        print("\nWARNING: Ollama is not available. To use Mistral or other models:")
        print("1. Install Ollama from https://ollama.ai/")
        print("2. Start the Ollama service")
        print("3. Pull the Mistral model: ollama pull mistral")
        print("\nContinuing with standard analysis methods...")
    
    # Perform comprehensive analysis and generate PDF report
    output_pdf = "data_analysis_report.pdf"
    results = analyzer.analyze_dataset(output_pdf)
    
    print(f"\nAnalysis complete! PDF report saved to: {output_pdf}")
    print("\nYou can now ask questions about the data using the ask() method:")
    print("Example: analyzer.ask('What are the key trends in this dataset?')")