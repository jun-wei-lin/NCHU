
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from dateutil.relativedelta import relativedelta
import csv


"""
請需下載以下套件
pip install tensorflow==2.1.0
pip install requests
pip install beautifulsoup4
pytorch 安裝較為複雜，需搭配cuda版本下不同指令，詳情請看以下網址
https://pytorch.org/get-started/locally/
安裝教學
https://stevehublog.medium.com/%E4%B8%80%E6%AD%A5%E4%B8%80%E6%AD%A5%E6%95%99%E6%82%A8%E5%9C%A8windows%E4%B8%8A%E5%AE%89%E8%A3%9Dpytorch%E8%88%87nvidia-cuda-a369eb88ee58

pip install keybert
pip install ckiptagger
pip install wordcloud
"""


#-----------------------------------------------------------------------

keyword="醬油" #要搜尋的關鍵字
peroid=3 #收集期間(單位:月)

#---------------------------------------------------------------------
#抓取資料
base_url = "https://www.ptt.cc"
url="https://www.ptt.cc/bbs/Gossiping/search?q="
url+=keyword

now_time=datetime.now()
now_time-=relativedelta(months=peroid)

end_searach=False
# 設定 cookies 避免 18 禁限制
cookies = {'over18': '1'}

article_webList=[]
articles=[]

# 尋找所有在時間區間內文章
while url:  # 當還有下一頁時繼續爬取
    web = requests.get(url, cookies=cookies)
    soup = BeautifulSoup(web.text, "html.parser")

    # 爬取當前頁面的標題
    titles = soup.find_all('div', class_='title')
    DateList = soup.find_all('div',class_ ='date')


    for i in range(len(titles)):
        if titles[i].find('a') is not None:  # 判斷是否有文章連結
            article_time=DateList[i].get_text()
            article_time=article_time.split('/')
            if len(article_time)==2:
                current_year=datetime.now().year
                article_time=datetime(current_year,int(article_time[0]),int(article_time[1]))
            else:
                article_time=datetime(int(article_time[0]),int(article_time[1]),int(article_time[2]))

            if article_time >now_time:

                print(titles[i].find('a').get_text())
                print(base_url + titles[i].find('a')['href'], end='\n\n')  # 文章連結
                article_webList.append(base_url + titles[i].find('a')['href'])
            else:
                end_searach=True
                break

    if end_searach:
        #找到該區間內所有文章，停止尋找
        break
    # 嘗試找到「上頁」按鈕
    btn_previous = soup.find('a', string="‹ 上頁")  # 查找文字為「‹ 上頁」的連結
    if btn_previous:
        url = base_url + btn_previous['href']  # 更新 URL 指向上頁
    else:
        break  # 如果沒有「上頁」按鈕，結束爬蟲


for url in article_webList:
    response = requests.get(url, cookies=cookies)
    soup = BeautifulSoup(response.text,"html.parser")
    ## 查找所有html 元素 抓出內容
    main_container = soup.find(id='main-container')
    # 把所有文字都抓出來
    all_text = main_container.text
    # 把整個內容切割透過 "-- " 切割成2個陣列
    pre_text = all_text.split('--')[0]

    # 把每段文字 根據 '\n' 切開
    texts = pre_text.split('\n')
    # 如果你爬多篇你會發現
    contents = texts[2:]
    # 內容
    content = ''.join(contents)
    content= "".join(content.split())

    articles.append(content)

path='articles.csv'
with open(path,'w',encoding='utf-8',newline='') as f:
  writer=csv.writer(f)
  for i in range(len(articles)):
    writer.writerow([articles[i]])


#---------------------------------------------------------------------------
#字詞分析

import os
os.environ["TF_USE_LEGACY_KERAS"]='1'
from ckiptagger import data_utils, construct_dictionary, WS, POS, NER
from sklearn.feature_extraction.text import CountVectorizer
from keras.layers import Bidirectional,LSTM
from keybert import KeyBERT
import csv
from wordcloud import WordCloud
import matplotlib.pyplot as plt

with open("articles.csv",'r',encoding='utf-8') as f:
  articles=csv.reader(f)
  articles=list(articles)

for i in range(len(articles)):
    articles[i]=articles[i][0]

#進行段詞處理
if not os.path.exists('./data'):
  data_utils.download_data_gdown("./")

#使用gpu
ws = WS("./data",disable_cuda=False)

keyword_analysis_result=[]

for article in articles:

  def token_func(text):
    return ws([text])[0]

  vectorizer = CountVectorizer(tokenizer=token_func, token_pattern=None)


  # Step 2: Initialize KeyBERT with the fitted CountVectorizer
  kw_model = KeyBERT()

  # Step 3: Extract keywords using the custom vectorizer
  keywords = kw_model.extract_keywords(article,vectorizer=vectorizer,top_n=30)

  keyword_analysis_result.append(keywords)


key_freq={}

for i in range(len(keyword_analysis_result)):
    for word,freq in keyword_analysis_result[i]:
        if word not in key_freq.keys():
            key_freq[word]=freq
        else:
            key_freq[word]+=freq

#去除常見字
freq_word=['不','的','沒','管','說','你','我','他','她','得','。','，','《','跟','被','那','人','中','「','?','!','有','有','就','了','個','是','是','之','去','嗎','很','在','再','啊','要','喵','到','大','八卦','這',
           '想','會','買','又','問','！','？','才','？','還','聽','幾','媒體','新聞網','剛剛','內文','網路','什麼','風向','怎麼','比較','一下']

for word in freq_word:
    if word in key_freq.keys():
        del key_freq[word]

key_freq=dict(sorted(key_freq.items(), key=lambda x:x[1],reverse=True)[:50])


font_path=os.path.abspath('.')
font_path=os.path.join(font_path,'mingliu.ttc')
wordcloud = WordCloud(width=800, height=400, background_color='white',font_path = font_path).generate_from_frequencies(key_freq)

# 顯示字雲
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # 隱藏座標軸
plt.title(f'Top 50 Word Cloud:{keyword}', fontsize=16)

plt.show()

input('案enter結束程式')