import os
import json
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from openai import OpenAI

# We use the existing NVIDIA client configuration to stay consistent.
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "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")
nv_client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = NVIDIA_API_KEY if NVIDIA_API_KEY else "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",
  timeout = 90.0, # Increased for complex LangGraph chains
  max_retries = 1
)

# Global store for the latest blog posts
LATEST_BLOG_POSTS = []

class AgentState(TypedDict):
    topics: List[str]
    blogs: List[dict]
    language: str

import time

def generate_topics(state: AgentState):
    """Generate current trending topics in dermatology using LLM."""
    print("[LangGraph Node] generate_topics started...")
    prompt = "You are a dermatology expert. Propose 3 trending topics in dermatology or skin care right now. Return ONLY a JSON list of 3 strings."
    
    topics = None
    for attempt in range(3):
        try:
            print(f"[LangGraph] Generating topics - Attempt {attempt + 1}/3...")
            response = nv_client.chat.completions.create(
                model="meta/llama-3.2-11b-vision-instruct",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.8,
                timeout=30.0  # Limit per-request wait to 30 seconds
            )
            content = response.choices[0].message.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
                
            topics = json.loads(content, strict=False)
            if not isinstance(topics, list) or len(topics) == 0:
                 topics = ["Phòng ngừa ung thư da", "Chăm sóc da mụn", "Lợi ích của Vitamin C"]
            print("[LangGraph] Topics generated successfully via LLM.")
            break
        except Exception as e:
            print(f"[LangGraph] Topic generation attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                sleep_time = 2 ** attempt
                print(f"[LangGraph] Sleeping for {sleep_time} seconds before retrying...")
                time.sleep(sleep_time)
            else:
                print("[LangGraph] All topic generation attempts failed. Using fallback topics.")
                topics = ["Tác hại của tia UV", "Nhận biết sớm ung thư da", "Quy trình skincare cơ bản"]
        
    return {"topics": topics}

def write_blogs(state: AgentState):
    """Write blog posts based on the fetched topics."""
    print("[LangGraph Node] write_blogs started...")
    topics = state.get("topics", [])
    lang = state.get("language", "Vietnamese")
    
    prompt = (
        f"Write 3 high-quality dermatology blog posts based on these exactly 3 topics: {topics}. "
        "Each post must have: \n"
        "- 'category' (1-2 words)\n"
        "- 'title'\n"
        "- 'desc' (2 sentences max)\n"
        "- 'content' (Full detailed article in HTML, at least 3 paragraphs, using <h3>, <p>, <ul> to format. DO NOT use markdown code blocks like ```html inside the string. Just raw HTML string).\n"
        "- 'visual_description' (A short 3-5 word English keyword phrase describing a professional stock photo for this topic, e.g., 'dermatology clinic', 'laser hair therapy', 'skin sunscreen protection'. DO NOT use any punctuation, commas, or quotes. Keep it strictly in English).\n"
        "Return ONLY a valid JSON array of exactly 3 objects. "
        f"The language of 'category', 'title', 'desc', and 'content' must be {lang}. However, the 'visual_description' field MUST ALWAYS be written in pure English and remain a very short keyword phrase (3-5 words) without punctuation to ensure clean URL generation."
    )

    blogs = None
    for attempt in range(3):
        try:
            print(f"[LangGraph] Writing blogs - Attempt {attempt + 1}/3...")
            response = nv_client.chat.completions.create(
                model="meta/llama-3.2-11b-vision-instruct",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=3000,
                temperature=0.7,
                timeout=60.0  # Limit per-request wait to 60 seconds
            )
            content = response.choices[0].message.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
                
            blogs = json.loads(content, strict=False)
            import urllib.parse
            import requests
            
            PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
            for blog in blogs:
                desc = blog.get('visual_description', 'dermatology clinic')
                clean_desc = desc.replace('"', '').replace("'", '').replace(',', '').replace('.', '').strip()
                
                img_url = None
                if PEXELS_API_KEY and PEXELS_API_KEY.strip():
                    try:
                        print(f"[Pexels] Searching for image with query: '{clean_desc}'...")
                        headers = {"Authorization": PEXELS_API_KEY}
                        # Call Pexels Search API
                        res = requests.get(
                            f"https://api.pexels.com/v1/search?query={urllib.parse.quote(clean_desc)}&per_page=1",
                            headers=headers,
                            timeout=10.0
                        )
                        if res.status_code == 200:
                            res_data = res.json()
                            if res_data.get('photos') and len(res_data['photos']) > 0:
                                img_url = res_data['photos'][0]['src'].get('large')
                                print(f"[Pexels] Found photo URL: {img_url}")
                        else:
                            print(f"[Pexels] Search failed with status code {res.status_code}")
                    except Exception as pe:
                        print(f"[Pexels] API connection error: {pe}")
                
                # Fallback to verified active Unsplash image if Pexels key is missing or search failed
                if not img_url:
                    print("[Pexels] Using verified active Unsplash fallback image.")
                    import random
                    verified_fallbacks = [
                        "https://images.unsplash.com/photo-1522337360788-8b13dee7a37e?q=80&w=800",
                        "https://images.unsplash.com/photo-1616391182219-e080b4d1043a?q=80&w=800",
                        "https://images.unsplash.com/photo-1556228720-195a672e8a03?q=80&w=800"
                    ]
                    img_url = random.choice(verified_fallbacks)
                
                blog['img'] = img_url
            print("[LangGraph] Blogs written and images generated successfully.")
            break
        except Exception as e:
            print(f"[LangGraph] Blog writing attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                sleep_time = 2 ** attempt
                print(f"[LangGraph] Sleeping for {sleep_time} seconds before retrying...")
                time.sleep(sleep_time)
            else:
                print("[LangGraph] All blog writing attempts failed. Using fallback blogs.")
                # Fallback with realistic-ish prompts
                blogs = [
                    { 
                        "category": "Cập nhật", 
                        "title": "Bảo vệ làn da của bạn", 
                        "desc": "Những phương pháp bảo vệ da trước ánh nắng mặt trời.", 
                        "content": "<h3>Tại sao phải bảo vệ da?</h3><p>Tia UV từ ánh nắng mặt trời là nguyên nhân chính gây ra lão hóa sớm và ung thư da. Việc sử dụng kem chống nắng hàng ngày giúp tạo lớp màng bảo vệ hiệu quả.</p><h3>Các bước cơ bản</h3><ul><li>Thoa kem chống nắng SPF 30+ trở lên.</li><li>Đeo kính râm và áo khoác khi ra ngoài.</li><li>Tránh ánh nắng trực tiếp từ 10h sáng đến 4h chiều.</li></ul>",
                        "img": "https://images.unsplash.com/photo-1522337360788-8b13dee7a37e?q=80&w=800" 
                    },
                    { 
                        "category": "Kiến thức", 
                        "title": "AI trong Y tế", 
                        "desc": "Skinderm AI giúp phát hiện ung thư da sớm hơn.", 
                        "content": "<h3>Trí tuệ nhân tạo (AI) hoạt động như thế nào?</h3><p>Công nghệ AI học hỏi từ hàng triệu hình ảnh y khoa để nhận diện các mô hình bất thường trên da. Tốc độ phân tích của AI chỉ mất vài giây với độ chính xác cao.</p><p>Skinderm AI sử dụng mô hình học sâu tiên tiến nhất, mang lại một phương pháp tầm soát nhanh chóng và tiện lợi ngay tại nhà.</p>",
                        "img": "https://images.unsplash.com/photo-1616391182219-e080b4d1043a?q=80&w=800" 
                    },
                    { 
                        "category": "Mẹo vặt", 
                        "title": "Dưỡng ẩm đúng cách", 
                        "desc": "Bí quyết dưỡng ẩm cho da mùa hanh khô hiệu quả.", 
                        "content": "<h3>Tầm quan trọng của việc dưỡng ẩm</h3><p>Nước chiếm tỷ lệ lớn trong cấu trúc da. Khi thiếu nước, da sẽ trở nên khô, nứt nẻ và dễ bị tổn thương bởi các tác nhân bên ngoài.</p><h3>Bí quyết cho bạn</h3><ul><li>Uống đủ 2 lít nước mỗi ngày.</li><li>Sử dụng kem dưỡng ẩm phù hợp với loại da.</li><li>Không rửa mặt bằng nước quá nóng.</li></ul>",
                        "img": "https://images.unsplash.com/photo-1556228720-195a672e8a03?q=80&w=800" 
                    }
                ]
                
    return {"blogs": blogs}

# --- Compile LangGraph Workflow ---
workflow = StateGraph(AgentState)
workflow.add_node("generate_topics", generate_topics)
workflow.add_node("write_blogs", write_blogs)

workflow.set_entry_point("generate_topics")
workflow.add_edge("generate_topics", "write_blogs")
workflow.add_edge("write_blogs", END)

explore_app = workflow.compile()

CACHE_FILE_PATH = os.path.join(os.environ.get("STORAGE_ROOT", "."), "explore_cache.json")

def load_cached_blogs():
    global LATEST_BLOG_POSTS
    if os.path.exists(CACHE_FILE_PATH):
        try:
            with open(CACHE_FILE_PATH, "r", encoding="utf-8") as f:
                LATEST_BLOG_POSTS = json.load(f)
            print(f"[LangGraph] Loaded {len(LATEST_BLOG_POSTS)} blogs from persistent cache: {CACHE_FILE_PATH}")
        except Exception as e:
            print(f"[LangGraph] Failed to load cached blogs: {e}")

def run_explore_workflow():
    global LATEST_BLOG_POSTS
    print("[LangGraph] Starting scheduled blog generation...")
    try:
        final_state = explore_app.invoke({"topics": [], "blogs": [], "language": "Vietnamese"})
        if final_state and "blogs" in final_state:
            LATEST_BLOG_POSTS = final_state["blogs"]
            print(f"[LangGraph] Generated {len(LATEST_BLOG_POSTS)} new blog posts successfully.")
            # Save to persistent cache
            try:
                os.makedirs(os.path.dirname(CACHE_FILE_PATH), exist_ok=True)
                with open(CACHE_FILE_PATH, "w", encoding="utf-8") as f:
                    json.dump(LATEST_BLOG_POSTS, f, ensure_ascii=False, indent=4)
                print(f"[LangGraph] Saved blogs to persistent cache: {CACHE_FILE_PATH}")
            except Exception as ce:
                print(f"[LangGraph] Failed to save blogs cache: {ce}")
    except Exception as e:
        print(f"[LangGraph] Workflow failed: {e}")

def get_latest_blogs():
    global LATEST_BLOG_POSTS
    if not LATEST_BLOG_POSTS:
        load_cached_blogs()
    return LATEST_BLOG_POSTS

# Load cache on import
load_cached_blogs()