import requests
import  warnings
warnings.filterwarnings('ignore')

headers = {
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36"
}


def get_comments(keywords , product_id):
    l = set()
    for i in range(10):
        try:
            url = "https://club.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98&productId={id}&score=0&sortType=5&page={page}&pageSize=10".format(
                id=product_id, page=str(i))
            text = requests.get(url, headers=headers, verify=False).text
            import re

            texts = re.findall(r'.*?content":(.*?"),', text.replace(" ", ""))

            if texts:
                for line in texts:
                    if line:
                        line = line.replace("$%&", "")  # 特殊
                        line = line.replace("&%$", "")
                        line = line.replace(r'\r\n', ' ')
                        line = line.replace(r"\n", ' ')
                        line = line.replace("{", "").replace("}", "")
                        line = line.replace("&hellip;", "")
                        line = re.sub('\s+', "", line)
                        line = line.strip().rstrip()  # 去两侧的空格
                        l.add(line)
        except Exception as e:
            pass

    with open("jd_comment_" +keywords + ".txt", 'a', encoding="utf-8") as file:
        for line in l:
            file.write(line + "\n")
    return l

from lxml import etree
import time
import random

def get_product_id(keywords):
    id_list = []
    for i in range(10):
        url = "https://search.jd.com/Search?keyword={keyword}&page={page}&enc=utf-8".format(keyword=keywords,
                                                                                            page=str(2 * i - 1))
        HTML = requests.get(url, headers=headers, verify=False).text
        mytree = etree.HTML(HTML)
        for item in mytree.xpath("//li[@class=\"gl-item\"]//i/@id"):
            temp = item.rsplit("_", 1)
            if temp:
                idx = temp[-1]
                id_list.append(idx)
        time.sleep(random.randint(1, 3))

        print("正在抓取下一页 --------------  ", i)

    return id_list


if __name__ == '__main__':
    # product_id = '100004770263'
    comments = []
    keywords = '平板'
    products = get_product_id( keywords )
    for product_id in products:
        comments.extend(get_comments( keywords , product_id))
        time.sleep(random.randint(1, 3))
