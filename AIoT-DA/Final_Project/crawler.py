import requests
from bs4 import BeautifulSoup
from datetime import datetime
from dateutil.relativedelta import relativedelta

def scrape_ptt(keyword, period):
    """
    爬取 PTT 八卦板文章內容。

    Args:
        keyword (str): 搜尋關鍵字
        period (int): 搜尋期間（單位：月）

    Returns:
        List[str]: 文章內容列表
    """
    base_url = "https://www.ptt.cc"
    url = f"{base_url}/bbs/Gossiping/search?q={keyword}"
    now_time = datetime.now() - relativedelta(months=period)
    cookies = {'over18': '1'}
    articles = []

    while url:
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
                                content = "".join(content.split())
                                # 限制內容長度
                                content = content[:2000]  # 限制為前 2000 字
                                articles.append(content)
                        else:
                            return articles
                    except ValueError:
                        # 日期解析錯誤，跳過該文章
                        continue

            # 找到「‹ 上頁」按鈕
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

    return articles
