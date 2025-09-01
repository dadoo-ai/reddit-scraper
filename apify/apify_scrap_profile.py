from apify_client import ApifyClient, ApifyClientAsync
from apify_client.client import DatasetClient
from dotenv import load_dotenv
import os
from typing import List, Dict
import json
from httpx import AsyncClient
load_dotenv()

# Initialize the ApifyClient with your API token


class ApifyProfileScraper:
    def __init__(self):
        self.client: ApifyClient = ApifyClient(os.getenv("APIFY_API_TOKEN"))
        self.dataset_id: str = None
        self.user_id: str = None    

    def scrape_profile(self,url:str):
        try:
            profile_id = url.split("/")[-1]
            run_input = {
                "startUrls": [{ "url": url }],
                "skipComments": False,
                "skipUserPosts": False,
                "skipCommunity": True,
                "ignoreStartUrls": False,
                "searchPosts": True,
                "searchComments": True,
                "searchCommunities": False,
                "searchUsers": False,
                "sort": "new",
                "includeNSFW": True,
                "maxItems": 200,
                "maxPostCount": 100,
                "maxComments": 50,
                "maxCommunitiesCount": 0,
                "maxUserCount": 0,
                "scrollTimeout": 40,
                "proxy": {
                    "useApifyProxy": True,
                    "apifyProxyGroups": ["RESIDENTIAL"],
                },
                "debugMode": False,
            }

            # Run the Actor and wait for it to finish
            run = self.client.actor("oAuCIx3ItNrs2okjQ").call(run_input=run_input)
            dataset: DatasetClient =self.client.dataset(run["defaultDatasetId"])
            items = json.loads(dataset.get_items_as_bytes(item_format="json",skip_empty=True))
            with open(f"results/users/profiles/profile_{profile_id}.json", "w", encoding="utf-8") as f:
                json.dump(items, f, indent=2, ensure_ascii=False)
            with open("results/users/profiles/done.txt", "a", encoding="utf-8") as log_file:
                log_file.write(f"{profile_id}\n")
        except Exception as e:
            print(f"Error: {e}")
            raise e


# APIFY_API_TOKEN = os.getenv('APIFY_API_TOKEN')
# profiles: List[Dict] = []

# class ApifyItemsCollector:
#     def __init__(self, folder_path: str):
#         self.folder_path = folder_path
#         self.profiles: List[Dict] = []
#         self.load_profiles(folder_path)

#     def load_profiles(self,folder_path: str)->List[Dict]:
#         files = os.listdir(folder_path)
#         for file in files:
#             if file.endswith(".json"):
#                 with open(os.path.join(folder_path, file), "r", encoding="utf-8") as f:
#                     self.profiles.append(json.load(f))
        
#     async def load_profile_items(self):
#         try:
#             print(self.profiles)
#             # url = f"https://api.apify.com/v2/datasets/{dataset_id}/items?token={APIFY_API_TOKEN}"
#             # headers = {
#             #     'Accept': 'application/json'
#             # }
#             # async with AsyncClient(http2=True, headers=headers, follow_redirects=True, timeout=30.0) as client:
#             #     response = await client.get(url, headers=headers)
#             #     result = response.json()
                
#             # with open("results/users/profiles/profile_{}.json", "w", encoding="utf-8") as f:
#             #     json.dump(result, f, indent=2, ensure_ascii=False)
#         except Exception as e:
#             print(f"Error: {e}")
#             raise e

