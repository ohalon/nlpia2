from ch05.fake_news_logistic_regression_title_len import *
# import numpy as np

try:
    assert len(__file__)
except (AssertionError, NameError):
    __file__ = "fake_news_histogram_title_len_red_blue.py"

isfake = y > 0
sns.histplot(X['title_len'][isfake], color='r', bins=list(range(0, 300, 5)))
sns.histplot(X['title_len'][~isfake], color='b', bins=list(range(0, 300, 5)))
image_filepath = CH05_IMAGES_DIR / (Path(__file__).name[:-2] + 'png')
print(f'Saving logistic regression plot to {image_filepath}')
plt.savefig(image_filepath)
plt.show()
