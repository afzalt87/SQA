import os
import time
import json
import base64
from dataclasses import dataclass
from typing import List, Dict, Optional
import pandas as pd
from io import BytesIO
from PIL import Image
import requests
import logging
from openai import OpenAI
import urllib.parse

logger = logging.getLogger(__name__)

@dataclass
class TrendingItem:
    """Class for representing a trending item from Yahoo TN API or Redux API"""
    position: int
    title: str
    description: Optional[str]
    image_url: str
    image_data: Optional[bytes] = None
    analysis_result: Optional[Dict] = None
    source: str = "TN_API"  # Track source: TN_API or REDUX_API

class YahooTrendAnalyzer:
    """Class for analyzing Yahoo trending module images and titles using dual APIs with web search capabilities"""
    
    def __init__(self, timeout=30, api_key=None):
        """Initialize the analyzer
        
        Args:
            timeout: Timeout for API requests in seconds
            api_key: OpenAI API key. If None, tries to use environment variable
        """
        self.timeout = timeout
        self.tn_api_url = "https://trending.search.yahoo.com:4443/tn"
        self.redux_api_url = "http://caketools.home-atadjine.gq2.ows.oath.cloud:4000/redux"
        self.items = []
        
        # Setting up OpenAI client
        if api_key:
            self.openai_client = OpenAI(api_key=api_key)
        else:
            api_key_env = os.environ.get("OPENAI_API_KEY")
            if not api_key_env:
                logger.warning("No OpenAI API key provided. Image analysis functionality will be disabled.")
                self.openai_client = None
            else:
                self.openai_client = OpenAI(api_key=api_key_env)
    
    def web_search(self, query: str) -> str:
        logger.info(f"Performing web search for: {query}")
        return f"Search results for '{query}': [This would contain actual search results from the API]"
    
    def fetch_from_tn_api(self, category="general", locale="en_us"):
        """Fetch trending items from Yahoo Trending TN API (20 items)
        
        Args:
            category: Category for trending items (default: general)
            locale: Locale for trending items (default: en_us)
            
        Returns:
            List of TrendingItem objects
        """
        params = {
            "category": category,
            "source": "p13n",
            "locale": locale
        }
        
        try:
            logger.info("Fetching trending items from TN API...")
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            }
            
            response = requests.get(self.tn_api_url, params=params, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            items_data = data.get("itemsInfo", {}).get("items", [])
            
            if not items_data:
                logger.warning("No items found in TN API response")
                return []
            
            logger.info(f"Found {len(items_data)} items in TN API response")
            
            trending_items = []
            
            # Process all items
            for idx, item_data in enumerate(items_data[:20]):
                try:
                    position = idx + 1  # Position starting from 1
                    title = item_data.get("display_term", item_data.get("search_term", "Unknown"))
                    description = item_data.get("raw_query", item_data.get("search_term", ""))
                    img_url = item_data.get("thumbnail", "")
                    img_data = None
                    
                    # Download image if URL is available
                    if img_url:
                        logger.debug(f"Downloading image from TN API: {img_url}")
                        try:
                            img_response = requests.get(img_url, headers=headers, timeout=self.timeout)
                            if img_response.status_code == 200:
                                img_data = img_response.content
                                logger.debug(f"Successfully downloaded image for TN API position {position}")
                            else:
                                logger.warning(f"Failed to download image from TN API: HTTP {img_response.status_code}")
                        except Exception as e:
                            logger.error(f"Error downloading image from TN API: {e}")
                    
                    item = TrendingItem(
                        position=position,
                        title=title,
                        description=description if description != title else None,
                        image_url=img_url,
                        image_data=img_data,
                        source="TN_API"
                    )
                    trending_items.append(item)
                    logger.info(f"Added TN API item {position}: {title}")
                    
                except Exception as e:
                    logger.error(f"Error processing TN API item: {e}")
                    continue
            
            logger.info(f"Successfully extracted {len(trending_items)} items from TN API")
            return trending_items
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching trending items from TN API: {e}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing TN API response: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error fetching from TN API: {e}")
            return []
    
    def fetch_from_redux_api(self, query="warriors"):
        """Fetch trending items from Redux API (6 production items)
        
        Args:
            query: Search query for Redux API
            
        Returns:
            List of TrendingItem objects
        """
        # Constructing the query string
        query_string = f'{{search(params:{{query:"{query}"}}){{query,intl,device,timestamp,gossip,data,snapshot}}}}'
        
        try:
            logger.info(f"Fetching trending items from Redux API for query: {query}")
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            }
        
            # Use GET request with query parameter
            params = {"query": query_string}
            response = requests.get(self.redux_api_url, params=params, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            
            trending_data = (data.get("data", {})
                        .get("search", {})
                        .get("data", {})
                        .get("trendingNow", {})
                        .get("trendingNowStory", {})
                        .get("data", {}))
            
            lists = trending_data.get("lists", [])
            
            if not lists:
                logger.warning("No trending items found in Redux API response")
                return []
            
            logger.info(f"Found {len(lists)} items in Redux API response")
            
            trending_items = []
        
            # Process Redux API items
            for idx, item_data in enumerate(lists):
                try:
                    # Position continues after TN API items (21-26)
                    position = idx + 21
                    title = item_data.get("text", "Unknown")
                    description = item_data.get("summary", "")
                    img_url = item_data.get("thumbnail", "")
                    img_data = None
                    
                    # Download image if URL is available
                    if img_url:
                        logger.debug(f"Downloading image from Redux API: {img_url}")
                        try:
                            img_response = requests.get(img_url, headers=headers, timeout=self.timeout)
                            if img_response.status_code == 200:
                                img_data = img_response.content
                                logger.debug(f"Successfully downloaded image for Redux API position {position}")
                            else:
                                logger.warning(f"Failed to download image from Redux API: HTTP {img_response.status_code}")
                        except Exception as e:
                            logger.error(f"Error downloading image from Redux API: {e}")
                    
                    item = TrendingItem(
                        position=position,
                        title=title,
                        description=description,
                        image_url=img_url,
                        image_data=img_data,
                        source="REDUX_API"
                    )
                    trending_items.append(item)
                    logger.info(f"Added Redux API item {position}: {title}")
                
                except Exception as e:
                    logger.error(f"Error processing Redux API item: {e}")
                    continue
            
            logger.info(f"Successfully extracted {len(trending_items)} items from Redux API")
            return trending_items
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching trending items from Redux API: {e}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing Redux API response: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error fetching from Redux API: {e}")
            return []

    
    def fetch_trending_module(self, query="news", category="general", locale="en_us"):
        """Fetch trending items from both TN API and Redux API
        
        Args:
            query: Search query (used for Redux API)
            category: Category for TN API trending items (default: general)
            locale: Locale for TN API trending items (default: en_us)
        """
        # Fetch from TN API (20 items)
        tn_items = self.fetch_from_tn_api(category=category, locale=locale)
        
        # Fetch from Redux API (6 items)
        redux_items = self.fetch_from_redux_api(query=query)
        
        # Combine all items
        self.items = tn_items + redux_items
        
        logger.info(f"Total trending items collected: {len(self.items)} (TN: {len(tn_items)}, Redux: {len(redux_items)})")
        return len(self.items) > 0
    
    def analyze_images(self, system_prompt=None, use_web_search=True):
        """Analyze images using GPT-4.1 with optional web search for context"""
        if not self.items:
            logger.warning("No items to analyze. Fetch trending module first.")
            return
        
        if not hasattr(self, 'openai_client') or self.openai_client is None:
            logger.warning("OpenAI API key not set. Skipping image analysis.")
            for item in self.items:
                item.analysis_result = {
                    "image_relevance": "N/A",
                    "text_relevance": "N/A", 
                    "image_quality": "N/A",
                    "image_integrity": "OpenAI API key not provided",
                    "justification": "Image analysis was skipped because no OpenAI API key was provided."
                }
            return
        
        if system_prompt is None:
            system_prompt = """You are an expert image analyst who specializes in image relevance and image
quality analysis. You have an eye for detail. When provided with a query, title, and
an image, you analyze the image thoroughly, do additional research if necessary to
understand the entity or subject, and then use your expertise to analyze the images
and provide accurate analysis.

You can use the web_search function to research topics, people, or events shown in the images
to provide more accurate and contextual analysis. Use web search when you need current information
about the subject matter or to verify facts.

You will be provided with an image or a link to an image along with a title and
accompanying text (description). Please research thoroughly and answer the following:
1. Is the image relevant to the title or description? Verify if the image and the text are about the same subject, topic, or entity. 
2. Does the image have any quality issues? Quality issues include blurred images,
Cropped images (particularly people's heads), grainy, noisy images etc. 
3. Is the image broken or not visible?
4. Does it look like the title and description are about the same news as one of the trends already examined? 
Please provide a justification for your decision for each image.
Output format:
Image relevance: [Relevant/Irrelevant]
Image quality: [Good/Bad quality issue description]
Image integrity: [None/Issue description]
Trend duplicate: [None/provide the previous duplicate] 
Justification: [Your detailed justification]
"""
    
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for current information about a topic, person, or event",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query",
                            },
                        },
                        "required": ["query"],
                    },
                }
            }
        ] if use_web_search else None
    
        for item in self.items:
            if not item.image_data:
                item.analysis_result = {
                    "image_relevance": "N/A",
                    "text_relevance": "N/A", 
                    "image_quality": "N/A",
                    "image_integrity": "Image not available",
                    "justification": "Image data could not be retrieved."
                }
                continue
            
            try:
                # Converting image to base64
                image = Image.open(BytesIO(item.image_data))
                buffered = BytesIO()
                image.save(buffered, format=image.format if image.format else "JPEG")
                base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
                
                # Determine image format (JPEG is default)
                img_format = image.format.lower() if image.format else "jpeg"
                
                # Creating user message with text and image
                user_text = f"Title: {item.title}\n"
                if item.description:
                    user_text += f"Description: {item.description}\n"
                user_text += f"Source: {item.source}\n"
                user_text += "Please analyze this image and provide your assessment. You can use web_search if you need current information about the subject."
                
                logger.info(f"Analyzing image for '{item.title[:30]}...' from {item.source}")
                
                try:
                    messages = [
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": user_text},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/{img_format};base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ]
                    
                    completion_args = {
                        "model": "gpt-4.1",
                        "messages": messages,
                        "temperature": 0.1,
                        "max_tokens": 1000
                    }
                    
                    if use_web_search:
                        completion_args["tools"] = tools
                        completion_args["tool_choice"] = "auto"
                    
                    completion = self.openai_client.chat.completions.create(**completion_args)
                    
                    response_message = completion.choices[0].message
                    
                    if use_web_search and response_message.tool_calls:
                        for tool_call in response_message.tool_calls:
                            if tool_call.function.name == "web_search":
                                function_args = json.loads(tool_call.function.arguments)
                                search_query = function_args.get("query")
                                
                                search_results = self.web_search(search_query)
                                
                                messages.append(response_message)
                                
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": search_results
                                })
                        
                        # Get the final response after tool use
                        final_completion = self.openai_client.chat.completions.create(
                            model="gpt-4.1",
                            messages=messages,
                            temperature=0.1,
                            max_tokens=1000
                        )
                        response_text = final_completion.choices[0].message.content
                    else:
                        # No tool calls, use the direct response
                        response_text = response_message.content
                
                    # Extract analysis results
                    lines = response_text.strip().split('\n')
                    result = {}
                    
                    for line in lines:
                        if line.startswith("Image relevance:"):
                            result["image_relevance"] = line.replace("Image relevance:", "").strip()
                        elif line.startswith("Text relevance:"):
                            result["text_relevance"] = line.replace("Text relevance:", "").strip()
                        elif line.startswith("Image quality:"):
                            result["image_quality"] = line.replace("Image quality:", "").strip()
                        elif line.startswith("Image integrity:"):
                            result["image_integrity"] = line.replace("Image integrity:", "").strip()
                        elif line.startswith("Trend duplicate:"):
                            result["trend_duplicate"] = line.replace("Trend duplicate:", "").strip()
                    
                    justification_index = response_text.find("Justification:")
                    if justification_index != -1:
                        result["justification"] = response_text[justification_index + len("Justification:"):].strip()
                    else:
                        result["justification"] = "No justification provided."
                    
                    item.analysis_result = result
                    
                except Exception as e:
                    logger.error(f"Error calling OpenAI API: {str(e)}")
                    item.analysis_result = {
                        "image_relevance": "Error",
                        "text_relevance": "Error", 
                        "image_quality": "Error",
                        "image_integrity": "Error",
                        "justification": f"Error analyzing image with OpenAI API: {str(e)}"
                    }
            
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error analyzing image for item '{item.title}': {e}")
                item.analysis_result = {
                    "image_relevance": "Error",
                    "text_relevance": "Error", 
                    "image_quality": "Error",
                    "image_integrity": "Error",
                    "justification": f"Error analyzing image: {str(e)}"
                }             
    
    def generate_report(self):
        """Generate a report of the analysis results"""
        if not self.items:
            logger.warning("No items to report. Fetch trending module and analyze images first.")
            return pd.DataFrame()
        
        # Create DataFrame
        report_data = []
        for item in self.items:
            if item.analysis_result:
                row = {
                    "Position": item.position,
                    "Title": item.title,
                    "Description": item.description or "",
                    "Image URL": item.image_url or "",
                    "Source": item.source,
                    "Image Relevance": item.analysis_result.get("image_relevance", "N/A"),
                    "Text Relevance": item.analysis_result.get("text_relevance", "N/A"),
                    "Image Quality": item.analysis_result.get("image_quality", "N/A"),
                    "Image Integrity": item.analysis_result.get("image_integrity", "N/A"),
                    "Trend Duplicate": item.analysis_result.get("trend_duplicate", "N/A"),
                    "Justification": item.analysis_result.get("justification", "N/A"),
                }
                report_data.append(row)
        
        df = pd.DataFrame(report_data)
        return df
    
    def save_images(self, directory="yahoo_trending_images"):
        """Save the images to a directory"""
        if not self.items:
            logger.warning("No items to save. Fetch trending module first.")
            return
        
        os.makedirs(directory, exist_ok=True)
        
        for item in self.items:
            if item.image_data:
                try:
                    image = Image.open(BytesIO(item.image_data))
                    filename = "".join(c if c.isalnum() or c in " .-_" else "_" for c in item.title)
                    filename = f"{item.position}_{item.source}_{filename[:50]}.{image.format.lower() if image.format else 'jpg'}"
                    filepath = os.path.join(directory, filename)
                    image.save(filepath)
                    logger.info(f"Saved image to {filepath}")
                except Exception as e:
                    logger.error(f"Error saving image for '{item.title}': {e}")
    
    def summary(self):
        """Generate a summary of the analysis results"""
        if not self.items or not all(item.analysis_result for item in self.items):
            logger.warning("No analysis results to summarize.")
            return {}
        
        # Separate summmaries by source
        tn_items = [item for item in self.items if item.source == "TN_API" and item.analysis_result]
        redux_items = [item for item in self.items if item.source == "REDUX_API" and item.analysis_result]
        
        def calculate_stats(items):
            if not items:
                return {
                    "Total Items": 0,
                    "Relevant Images (%)": 0,
                    "Relevant Text (%)": 0,
                    "Good Quality Images (%)": 0,
                    "No Integrity Issues (%)": 0,
                }
            
            total = len(items)
            relevant_images = sum(1 for item in items 
                                if item.analysis_result.get("image_relevance", "").lower() == "relevant")
            relevant_text = sum(1 for item in items 
                              if item.analysis_result.get("text_relevance", "").lower() == "relevant")
            good_quality = sum(1 for item in items 
                             if "good" in item.analysis_result.get("image_quality", "").lower())
            no_integrity_issues = sum(1 for item in items 
                                   if "none" in item.analysis_result.get("image_integrity", "").lower())
            
            return {
                "Total Items": total,
                "Relevant Images (%)": round(relevant_images / total * 100 if total else 0, 2),
                "Relevant Text (%)": round(relevant_text / total * 100 if total else 0, 2),
                "Good Quality Images (%)": round(good_quality / total * 100 if total else 0, 2),
                "No Integrity Issues (%)": round(no_integrity_issues / total * 100 if total else 0, 2),
            }
        
        # Overall stats
        all_stats = calculate_stats(self.items)
        tn_stats = calculate_stats(tn_items)
        redux_stats = calculate_stats(redux_items)
        
        summary = {
            "Overall": all_stats,
            "TN API": tn_stats,
            "Redux API": redux_stats
        }
        
        return summary


def run_trending_searches(query="news", api_key=None, use_web_search=True):
    """Run the complete trending searches analysis
    
    Args:
        query: Search query (used for Redux API)
        api_key: OpenAI API key (optional)
        use_web_search: Whether to enable web search during analysis
    
    Returns:
        Tuple of (List of issues found, DataFrame with detailed report)
    """
    analyzer = None
    try:
        analyzer = YahooTrendAnalyzer(api_key=api_key)
        
        logger.info("Fetching trending items from both APIs...")
        success = analyzer.fetch_trending_module(query=query)
        if not success:
            logger.error("Failed to fetch trending items.")
            return [], pd.DataFrame()
        
        logger.info(f"Found {len(analyzer.items)} total trending items.")
        
        logger.info(f"Analyzing images (web search: {'enabled' if use_web_search else 'disabled'})...")
        analyzer.analyze_images(use_web_search=use_web_search)
        
        issues = []
        for item in analyzer.items:
            if item.analysis_result:
                if item.analysis_result.get("image_relevance", "").lower() != "relevant":
                    issues.append({
                        "error_type": "trending_image_relevance",
                        "query": query,
                        "module": "trendingNow",
                        "offending_string": item.title,
                        "matched_token": "image_irrelevant",
                        "category": "image_mismatch",
                        "is_dead": "no",
                        "position": item.position,
                        "source": item.source,
                        "justification": item.analysis_result.get("justification", "")
                    })
                
                if "bad" in item.analysis_result.get("image_quality", "").lower():
                    issues.append({
                        "error_type": "trending_image_quality",
                        "query": query,
                        "module": "trendingNow",
                        "offending_string": item.title,
                        "matched_token": "poor_quality",
                        "category": "image_quality",
                        "is_dead": "no",
                        "position": item.position,
                        "source": item.source,
                        "justification": item.analysis_result.get("justification", "")
                    })
                
                if item.analysis_result.get("image_integrity", "").lower() not in ["none", "n/a"]:
                    issues.append({
                        "error_type": "trending_image_integrity",
                        "query": query,
                        "module": "trendingNow",
                        "offending_string": item.title,
                        "matched_token": "integrity_issue",
                        "category": "image_integrity",
                        "is_dead": "no",
                        "position": item.position,
                        "source": item.source,
                        "justification": item.analysis_result.get("justification", "")
                    })
        
        # Generate the detailed report
        report_df = analyzer.generate_report()
        
        # Log summary statistics
        summary = analyzer.summary()
        logger.info("Analysis Summary:")
        for source, stats in summary.items():
            logger.info(f"{source}: {stats}")
        
        return issues, report_df
        
    except Exception as e:
        logger.error(f"Error during trending searches analysis: {e}")
        return [], pd.DataFrame()