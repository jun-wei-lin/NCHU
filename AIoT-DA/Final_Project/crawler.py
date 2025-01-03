import requests
from bs4 import BeautifulSoup
from datetime import datetime
from dateutil.relativedelta import relativedelta
import re

def scrape_ptt(keyword, period, max_articles=100):
    """
    爬取 PTT 八卦板文章內容，並進行文本清洗與限制文章數量。

    
    Args:
        keyword (str): 搜尋關鍵字
        period (int): 搜尋期間（單位：月）
        max_articles (int): 最大文章數量限制

    Returns:
       List[Tuple[str, str]]: 清洗後的文章內容列表、每篇文章的內容和原文連結
    """
    base_url = "https://www.ptt.cc"
    url = f"{base_url}/bbs/Gossiping/search?q={keyword}"
    now_time = datetime.now() - relativedelta(months=period)
    cookies = {'over18': '1'}
    articles = []

    while url and len(articles) < max_articles:
        try:
            web = requests.get(url, cookies=cookies)
            web.raise_for_status()
            soup = BeautifulSoup(web.text, "html.parser")
            titles = soup.find_all('div', class_='title')
            dates = soup.find_all('div', class_='date')

            for i in range(len(titles)):
                if titles[i].find('a'):  # 確保有連結
                    date_text = dates[i].get_text().strip()
                    try:
                        article_date = datetime(datetime.now().year, *map(int, date_text.split('/')))
                        if article_date >= now_time:
                            link = base_url + titles[i].find('a')['href']
                            # 抓取文章內容
                            article_response = requests.get(link, cookies=cookies)
                            article_response.raise_for_status()
                            article_soup = BeautifulSoup(article_response.text, "html.parser")
                            content_element = article_soup.find(id="main-container")
                            if content_element:
                                content = content_element.text.split("--")[0]
                                # 清洗文本
                                content = re.sub(r"[^\w\s]", "", content)  # 移除標點符號
                                content = re.sub(r"\s+", " ", content)  # 合併多餘空格
                                content = re.sub(r"(作者|看板|標題|時間).*?:", "", content)  # 移除無效元數據
                                content = content.strip()  # 去掉首尾空格
                                
                                # 限制內容長度
                                content = content[:1500]  # 限制文章字符長度，避免超長文本
                           
                                # 只保留包含關鍵字的文章
                                if keyword in content:
                                    articles.append(content)
                                    
                                articles_with_links.append((content, link))
                                
                        else:
                            return articles_with_links
                    except ValueError:
                        continue  # 日期解析錯誤，跳過該文章

            next_page = soup.find('a', string="‹ 上頁")
            if next_page and 'href' in next_page.attrs:
                url = base_url + next_page['href']
            else:
                break

        except requests.RequestException as e:
            print(f"HTTP 請求錯誤：{e}")
            break
        except Exception as e:
            print(f"其他錯誤：{e}")
            break

    return articles[:max_articles]  # 確保返回的文章數量不超過限制
