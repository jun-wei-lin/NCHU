import requests
from bs4 import BeautifulSoup
from datetime import datetime
from dateutil.relativedelta import relativedelta

def scrape_ptt(keyword, period):
    """
    爬取 PTT 八卦板中包含指定關鍵字的文章。
    
    Args:
        keyword (str): 搜尋的關鍵字
        period (int): 搜尋期間（月）
    
    Returns:
        list: 包含文章連結的列表
    """
    base_url = "https://www.ptt.cc"
    url = f"{base_url}/bbs/Gossiping/search?q={keyword}"
    now_time = datetime.now() - relativedelta(months=period)
    cookies = {'over18': '1'}
    article_list = []

    while url:
        web = requests.get(url, cookies=cookies)
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
                        article_list.append(link)
                    else:
                        return article_list
                except ValueError:
                    continue  # 遇到無效日期時跳過

        next_page = soup.find('a', string="‹ 上頁")
        if next_page and 'href' in next_page.attrs:
            url = base_url + next_page['href']
        else:
            url = None

    return article_list
