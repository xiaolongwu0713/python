import matplotlib
matplotlib.use('Qt5Agg')
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

#mean_hunger = 5
#samples_per_day = 100
#n_days = 10000
#samples = np.random.normal(loc=mean_hunger, scale=1.0, size=(n_days, samples_per_day))
#daily_maxes = np.max(samples, axis=1)

# # gumbel的通用PDF公式见维基百科
# def gumbel_pdf(prob,loc,scale):
#     z = (prob-loc)/scale
#     return np.exp(-z-np.exp(-z))/scale

# def plot_maxes(daily_maxes):
#     probs,bins,_ = plt.hist(daily_maxes,density=True,bins=100)
#     print(f"==>> probs: {probs}") # 每个bin的概率
#     print(f"==>> bins: {bins}") # 即横坐标的tick值
#     print(f"==>> _: {_}")
#     print(f"==>> probs.shape: {probs.shape}") # (100,)
#     print(f"==>> bins.shape: {bins.shape}") # (101,)
#     plt.xlabel('Volume')
#     plt.ylabel('Probability of Volume being daily maximum')
#
#     # 基于直方图，下面拟合出它的曲线。
#
#     (fitted_loc, fitted_scale), _ = curve_fit(gumbel_pdf, bins[:-1],probs)
#     print(f"==>> fitted_loc: {fitted_loc}")
#     print(f"==>> fitted_scale: {fitted_scale}")
#     #curve_fit用于曲线拟合，doc：https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
#     #比如我们要拟合函数y=ax+b里的参数a和b，a和b确定了，这个函数就确定了，为了拟合这个函数，我们需要给curve_fit()提供待拟合函数的输入和输出样本
#     #所以curve_fit()的三个入参是：1.待拟合的函数（要求该函数的第一个入参是输入，后面的入参是要拟合的函数的参数）、2.样本输入、3.样本输出
#     #返回的是拟合好的参数，打包在元组里
#     # 其他教程：https://blog.csdn.net/guduruyu/article/details/70313176
#     plt.plot(bins, gumbel_pdf(bins, fitted_loc, fitted_scale))
#

# for example: there are 7 categories
n_cats = 7
cats = np.arange(n_cats)
probs = np.random.randint(low=1, high=20, size=n_cats)
probs = probs / sum(probs) # original probs
logits = np.log(probs)
n_samples = 100000

def sample_gumbel(logits):
    noise = np.random.gumbel(size=len(logits))
    sample = np.argmax(logits+noise)
    return sample
gumbel_samples = [sample_gumbel(logits) for _ in range(n_samples)]

def sample_uniform(logits):
    noise = np.random.uniform(size=len(logits))
    sample = np.argmax(logits+noise)
    return sample
uniform_samples = [sample_uniform(logits) for _ in range(n_samples)]

def sample_normal(logits):
    noise = np.random.normal(size=len(logits))
    sample = np.argmax(logits+noise)
    return sample
normal_samples = [sample_normal(logits) for _ in range(n_samples)]

def plot_probs():
    plt.bar(cats, probs)
    plt.xlabel("Category")
    plt.ylabel("Probability")
def plot_estimated_probs(samples,ylabel=''):
    n_cats = np.max(samples)+1
    estd_probs,_,_ = plt.hist(samples,bins=np.arange(n_cats+1),align='left',edgecolor='white',density=True)
    plt.xlabel('Category')
    plt.ylabel(ylabel+'Estimated probability')
    return estd_probs

fig,ax=plt.subplots(1,4)
plt.subplot(1,4,1)
plot_probs()
plt.subplot(1,4,2)
gumbel_estd_probs = plot_estimated_probs(gumbel_samples,'Gumbel ')
plt.subplot(1,4,3)
normal_estd_probs = plot_estimated_probs(normal_samples,'Normal ')
plt.subplot(1,4,4)
uniform_estd_probs = plot_estimated_probs(uniform_samples,'Uniform ')
plt.tight_layout()
#
# print('Original probabilities:\t\t',end='')
# print_probs(probs)
# print('Gumbel Estimated probabilities:\t',end='')
# print_probs(gumbel_estd_probs)
# print('Normal Estimated probabilities:\t',end='')
# print_probs(normal_estd_probs)
# print('Uniform Estimated probabilities:',end='')
# print_probs(uniform_estd_probs)
