from ch05.fake_news_logistic_regression_title_len import *
import numpy as np

try:
    assert len(__file__)
except (AssertionError, NameError):
    __file__ = "fake_news_logistic_regression_title_len_red_blue.py"

data = df[['title_len', 'isfake']].sample(250)
data['proba'] = model.predict_proba(data[['title_len']])[:, 1]
data = data.sort_values('proba')
isfake = data['isfake'] > 0
above50 = data['proba'] > .5

# ax = sns.regplot(x=data['title_len'][~isfake], y=data['isfake'][~isfake], logistic=True, color=color)
ax = sns.scatterplot(
    x=data['title_len'][~isfake], y=data['isfake'][~isfake],
    s=60, color='darkblue', alpha=.4)
ax = sns.scatterplot(
    x=data['title_len'][isfake], y=data['isfake'][isfake],
    s=60, color='darkred', alpha=.45)
# sns.lineplot(x=data['title_len'][isfake], y=data['title_len'][isfake].apply(estimator), color='red')
color = np.array(['r'] * len(data))


sns.lineplot(
    x=data['title_len'][~above50], y=data['proba'][~above50],
    linewidth=4, color='darkblue', alpha=.4, legend=False)
sns.lineplot(
    x=data['title_len'][above50], y=data['proba'][above50],
    linewidth=4, color='darkred', alpha=.45, legend=False)
plt.xlabel('Title Length (chars)')
plt.ylabel('Probability of Being Fake News')

image_filepath = CH05_IMAGES_DIR / (Path(__file__).name[:-2] + 'png')
print(f'Saving logistic regression plot to {image_filepath}')
plt.savefig(image_filepath)
plt.show()
