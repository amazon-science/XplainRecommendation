"""
Amazon Bedrock LLM Integration for RL-GraphRetriever

Replaces OpenAI GPT-3.5-turbo with Claude 3 Haiku for explanation generation.
Based on the approach from feedback_tone_analyzer.ipynb
"""

import boto3
import json
from pathlib import Path
from configparser import ConfigParser
from typing import Dict, Any, List, Optional
import time


class BedrockLLM:
    """Amazon Bedrock LLM wrapper for explanation generation"""
    
    def __init__(
        self,
        model_id: str = "anthropic.claude-3-haiku-20240307-v1:0",
        region: str = "us-east-1",
        aws_profile: str = "default",
        max_tokens: int = 1000,
        temperature: float = 0.3,
    ):
        """
        Initialize Bedrock LLM client
        
        Parameters
        ----------
        model_id : str
            Bedrock model ID (default: Claude 3 Haiku)
        region : str
            AWS region where Bedrock is available
        aws_profile : str
            AWS profile name from ~/.aws/credentials
        max_tokens : int
            Maximum tokens for generation
        temperature : float
            Sampling temperature (0-1)
        """
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Load credentials from ~/.aws/credentials
        creds = self._load_aws_credentials(aws_profile)
        
        # Create Bedrock client
        session = boto3.Session(
            aws_access_key_id=creds['aws_access_key_id'],
            aws_secret_access_key=creds['aws_secret_access_key'],
            aws_session_token=creds.get('aws_session_token')
        )
        
        self.bedrock_runtime = session.client(
            service_name='bedrock-runtime',
            region_name=region
        )
        
        print(f"✓ Bedrock LLM initialized")
        print(f"  Model: {self.model_id}")
        print(f"  Region: {region}")
    
    def _load_aws_credentials(self, profile_name: str = 'default') -> Dict[str, str]:
        """Load AWS credentials from ~/.aws/credentials"""
        credentials_path = Path.home() / '.aws' / 'credentials'
        
        if not credentials_path.exists():
            raise FileNotFoundError(f"AWS credentials file not found at {credentials_path}")
        
        config = ConfigParser()
        config.read(credentials_path)
        
        if profile_name not in config.sections():
            available_profiles = config.sections()
            raise ValueError(f"Profile '{profile_name}' not found. Available: {available_profiles}")
        
        credentials = {
            'aws_access_key_id': config[profile_name]['aws_access_key_id'],
            'aws_secret_access_key': config[profile_name]['aws_secret_access_key']
        }
        
        if 'aws_session_token' in config[profile_name]:
            credentials['aws_session_token'] = config[profile_name]['aws_session_token']
        
        return credentials
    
    def generate_explanation(
        self,
        user_profile: str,
        item_profile: str,
        item_title: str,
        retrieved_context: str,
        system_prompt: Optional[str] = None,
        dataset: str = "amazon"
    ) -> str:
        """
        Generate explanation using Bedrock LLM
        
        Parameters
        ----------
        user_profile : str
            User's profile/history
        item_profile : str
            Item's profile/description
        item_title : str
            Item title
        retrieved_context : str
            Retrieved graph paths and semantic context
        system_prompt : str, optional
            Custom system prompt (if None, uses default)
        dataset : str
            Dataset name (amazon, yelp, google) for prompt customization
        
        Returns
        -------
        explanation : str
            Generated explanation
        """
        # Construct dataset-optimized prompts
        if dataset == "amazon":
            task_desc = "Given the book title, book profile, and user profile, explain why the user would buy this book. Be specific about book content and user interests. Use exactly 40-50 words."
            item_type = "Book"
        elif dataset == "yelp":
            task_desc = "Given the business title, business profile, and user profile, explain why the user would enjoy this business. Focus on food, atmosphere, and user preferences. Use exactly 40-50 words."
            item_type = "Business"
        elif dataset == "google":
            task_desc = "Given the business title, business profile, and user profile, explain why the user would visit this business. Be concise and direct. Use exactly 25-30 words."
            item_type = "Business"
        else:
            task_desc = "Explain why the user would be interested in this item within 50 words."
            item_type = "Item"
        
        user_message = f"""{task_desc}
{item_type} title: {item_title}
{item_type} profile: {item_profile}
User profile: {user_profile}

### {retrieved_context}

### Explanation:"""
        
        # Prepare request body for Claude
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [
                {
                    "role": "user",
                    "content": user_message
                }
            ]
        }
        
        # Add system prompt if provided
        if system_prompt:
            request_body["system"] = system_prompt
        
        try:
            # Invoke model
            response = self.bedrock_runtime.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body)
            )
            
            # Parse response
            response_body = json.loads(response['body'].read())
            explanation = response_body['content'][0]['text']
            
            return explanation.strip()
            
        except Exception as e:
            print(f"Error generating explanation: {str(e)}")
            return ""
    
    def batch_generate_explanations(
        self,
        batch_data: List[Dict[str, str]],
        delay: float = 0.1
    ) -> List[str]:
        """
        Generate explanations for a batch of user-item pairs
        
        Parameters
        ----------
        batch_data : list of dict
            Each dict contains: user_profile, item_profile, item_title, 
            retrieved_context, dataset
        delay : float
            Delay between requests to avoid rate limiting
        
        Returns
        -------
        explanations : list of str
            Generated explanations
        """
        explanations = []
        
        for data in batch_data:
            explanation = self.generate_explanation(
                user_profile=data['user_profile'],
                item_profile=data['item_profile'],
                item_title=data['item_title'],
                retrieved_context=data['retrieved_context'],
                dataset=data.get('dataset', 'amazon')
            )
            explanations.append(explanation)
            
            # Rate limiting
            if delay > 0:
                time.sleep(delay)
        
        return explanations
    
    def evaluate_explanation_quality(
        self,
        prediction: str,
        reference: str,
        system_prompt: Optional[str] = None
    ) -> float:
        """
        Use LLM to evaluate explanation quality (for reward function)
        
        Parameters
        ----------
        prediction : str
            Generated explanation
        reference : str
            Ground truth explanation
        system_prompt : str, optional
            Evaluation criteria prompt
        
        Returns
        -------
        score : float
            Quality score (0-10)
        """
        if system_prompt is None:
            system_prompt = """You are an expert evaluator of recommendation explanations. 
Rate the quality of the generated explanation compared to the reference on a scale of 0-10.
Consider: relevance, coherence, informativeness, and similarity to the reference.
Respond with ONLY a number between 0 and 10."""
        
        eval_prompt = json.dumps({
            "prediction": prediction,
            "reference": reference
        })
        
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 10,
            "temperature": 0.0,
            "system": system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": eval_prompt
                }
            ]
        }
        
        try:
            response = self.bedrock_runtime.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body)
            )
            
            response_body = json.loads(response['body'].read())
            score_text = response_body['content'][0]['text'].strip()
            
            # Extract numeric score
            try:
                score = float(score_text)
                return max(0.0, min(10.0, score))  # Clamp to [0, 10]
            except ValueError:
                # Try to extract first number
                import re
                numbers = re.findall(r'\d+\.?\d*', score_text)
                if numbers:
                    return float(numbers[0])
                return 5.0  # Default middle score
                
        except Exception as e:
            print(f"Error evaluating explanation: {str(e)}")
            return 5.0  # Default score on error


# Example usage
if __name__ == "__main__":
    # Initialize LLM
    llm = BedrockLLM(
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        region="us-east-1",
        aws_profile="default"
    )
    
    # Test explanation generation
    explanation = llm.generate_explanation(
        user_profile="User enjoys fantasy and science fiction books",
        item_profile="Epic fantasy novel with dragons and magic",
        item_title="The Dragon's Crown",
        retrieved_context="Similar users who enjoy fantasy also bought this book",
        dataset="amazon"
    )
    
    print(f"Generated explanation: {explanation}")
    
    # Test evaluation
    score = llm.evaluate_explanation_quality(
        prediction=explanation,
        reference="This book matches the user's interest in fantasy and magic"
    )
    
    print(f"Quality score: {score}/10")
