import requests
from bs4 import BeautifulSoup
from datetime import datetime
from dateutil.relativedelta import relativedelta
import re
import pandas as pd
from time import sleep

def scrape_ptt(keyword, period, max_articles=100):
    """
    爬取 PTT 八卦板文章內容，並進行文本清洗與限制文章數量。

    Args:
        keyword (str): 搜尋關鍵字
        period (int): 搜尋期間（單位：月）
        max_articles (int): 最大文章數量限制

    Returns:
        List[str]: 清洗後的文章內容列表
        List[str]: 對應的文章連結列表  **(新增描述)**
    """
    base_url = "https://www.ptt.cc"
    url = f"{base_url}/bbs/Gossiping/search?q={keyword}"
    now_time = datetime.now() - relativedelta(months=period)
    cookies = {'over18': '1'}
    articles = []
    links = []  # **(新增) 初始化連結列表**
    
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
                            link = base_url + titles[i].find('a')['href']  # **(保持連結提取邏輯)**
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
                                    articles.append(content)  # **(保持原邏輯)**
                                    links.append(link)  # **(新增：保存連結至列表)**
                        else:
                            return articles, links  # **(更新返回兩個列表)**
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

    return articles[:max_articles], links[:max_articles]  # **(更新：返回兩個列表且限制數量)**


def scrape_keyword_trends(keyword, on_progress=None, timeout=10):
    """
    爬取 PTT 八卦板關鍵字的每月文章數據，用於趨勢分析。

    Args:
        keyword (str): 搜尋關鍵字
        on_progress (function): 進度回調函數，接受一個字符串參數
        timeout (int): 每個請求的超時時間（秒）

    Returns:
        pd.DataFrame: 包含月份和文章數的 DataFrame
    """
    base_url = "https://www.ptt.cc"
    url = f"{base_url}/bbs/Gossiping/search?q={keyword}"
    cookies = {'over18': '1'}
    trends = {}
    start_time = datetime.now() - relativedelta(years=1)  # 起始時間：過去一年
    page_count = 0  # 記錄頁數

    while url:
        try:
            max_retries = 3
            retries = max_retries
            while retries > 0:
                try:
                    response = requests.get(url, cookies=cookies, timeout=timeout)
                    response.raise_for_status()
                    break
                except requests.RequestException as e:
                    retries -= 1
                    if retries == 0:
                        if on_progress:
                            on_progress(f"HTTP 請求錯誤，已重試 {max_retries} 次仍失敗：{e}")
                        return pd.DataFrame(columns=["month", "value"])
                    sleep(1)

            soup = BeautifulSoup(response.text, "html.parser")
            titles = soup.find_all('div', class_='title')
            dates = soup.find_all('div', class_='date')

            for i in range(len(titles)):
                if titles[i].find('a'):
                    date_text = dates[i].get_text().strip()
                    if not re.match(r"^\d{1,2}/\d{1,2}$", date_text):
                        if on_progress:
                            on_progress(f"日期格式錯誤，跳過文章：{date_text}")
                        continue
                    try:
                        article_date = datetime(datetime.now().year, *map(int, date_text.split('/')))
                        if article_date > datetime.now():
                            article_date = article_date.replace(year=article_date.year - 1)
                        if article_date < start_time:
                            if on_progress:
                                on_progress("已超過設定的起始時間範圍，停止爬取。")
                            return pd.DataFrame(list(trends.items()), columns=["month", "value"])
                        month_str = article_date.strftime("%Y-%m")
                        trends[month_str] = trends.get(month_str, 0) + 1
                    except ValueError as e:
                        if on_progress:
                            on_progress(f"日期解析失敗，跳過文章：{date_text} ({e})")
                        continue

            page_count += 1
            if on_progress:
                on_progress(f"目前已爬取第 {page_count} 頁數據，範圍：{list(trends.keys())[-5:]}")

            next_page = soup.find('a', string="‹ 上頁")
            if next_page and 'href' in next_page.attrs:
                url = base_url + next_page['href']
            else:
                break

        except Exception as e:
            if on_progress:
                on_progress(f"其他錯誤：{e}")
            break

    trend_data = pd.DataFrame(list(trends.items()), columns=["month", "value"])
    trend_data["month"] = pd.to_datetime(trend_data["month"], errors="coerce")
    trend_data = trend_data.dropna()
    trend_data.sort_values(by="month", inplace=True)
    trend_data = trend_data.drop_duplicates(subset="month")

    if on_progress:
        on_progress(f"爬取完成！數據範圍：{trend_data['month'].min()} 到 {trend_data['month'].max()}")
        on_progress(f"數據樣本：\n{trend_data.head()}")

    return trend_data

def scrape_user_behavior(keyword, period, on_progress=None):
    """
    爬取 PTT 八卦板的用戶行為數據，包含作者與回文數。

    Args:
        keyword (str): 搜尋關鍵字
        period (int): 搜尋期間（單位：月）
        on_progress (function): 回調函數，用於顯示進度訊息

    Returns:
        List[Dict]: 每篇文章的用戶行為數據列表
    """
    base_url = "https://www.ptt.cc"
    url = f"{base_url}/bbs/Gossiping/search?q={keyword}"
    now_time = datetime.now() - relativedelta(months=period)
    cookies = {'over18': '1'}
    user_data = []
    article_count = 0
    earliest_date = None  # 新增：記錄最早日期

    while url:
        try:
            # 爬取頁面
            response = requests.get(url, cookies=cookies)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            titles = soup.find_all('div', class_='title')
            dates = soup.find_all('div', class_='date')

            for i in range(len(titles)):
                if titles[i].find('a'):
                    date_text = dates[i].get_text().strip()
                    try:
                        # 將文章日期轉為 datetime 格式
                        article_date = datetime(datetime.now().year, *map(int, date_text.split('/')))
                        if article_date >= now_time:
                            link = base_url + titles[i].find('a')['href']

                            # 爬取文章詳情
                            article_response = requests.get(link, cookies=cookies)
                            article_response.raise_for_status()
                            article_soup = BeautifulSoup(article_response.text, "html.parser")
                            author = article_soup.find('span', class_='article-meta-value')
                            replies = article_soup.find_all('span', class_='push-tag')

                            # 統計回文數量
                            reply_count = len(replies)

                            # 儲存用戶行為數據
                            user_data.append({
                                "author": author.text if author else "匿名用戶",
                                "reply_count": reply_count,
                                "date": article_date.strftime('%Y-%m-%d'),
                            })

                            # 更新最早日期
                            if earliest_date is None or article_date < earliest_date:
                                earliest_date = article_date

                            article_count += 1

                            # 更新進度
                            if on_progress:
                                on_progress(f"已爬取 {article_count} 篇文章，日期範圍：{earliest_date.strftime('%Y-%m-%d')} 至 {datetime.now().strftime('%Y-%m-%d')}")

                    except ValueError:
                        continue

            # 找到下一頁
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

    return user_data

