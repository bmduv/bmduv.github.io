Data comes from the following distribution - P(X, Y). If we have access to P(X, Y), we can use Bayes Optimal Classifier where we can predict most likely label for x - $argmax_{y}$P(y | x).

Many if not most of supervised learning is estimating P(X, Y)
P(X, Y) = P(Y | X) P(X) (Discriminative) = P(X | Y) P(Y) (Generative)

## Maximum Likelihood Estimation: 
1. Make an explicit modeling assumption about what type of distribution data is sampled from.
2. Set parameters of this distribution so data you observed is as likely as possible.

Natural assumption about coin toss is that distribution of observed outcomes is binomial distribution - has two parameters $n$ and $\theta$ and it captures distribution of $n$ independent Bernoulli (binary) random events that have positive outcome with proabability $\theta$. n is number of coin tosses, and $\theta$ is probability of coin coming up heads. 
$$
\begin{align} P(D\mid \theta) &= \begin{pmatrix} n_H + n_T \\ n_H \end{pmatrix} \theta^{n_H} (1 - \theta)^{n_T}, \end{align}
$$
Computes probability we would observe exactly n_H heads, n_T tails, if coin was tossed n=n_H+n_T times and probability of coming up heads is $\theta$

MLE Principle: Find $\hat{\theta}$ to maximized likelihood of data, $P(D; \theta)$:
$$ 
\begin{align} \hat{\theta}_{MLE} &= \operatorname*{argmax}_{\theta} \,P(D ; \theta) \end{align}
$$
1. Plug in all terms for distribution , and take $\log$ of function.
2. Compute its derivative and equate with zero. Taking the log of likelihood does not change its maximum (as log is monotonic function, and likelihood positive), but turns all products into sums which are much easier to deal with for differentiation.
$$
\begin{align} \hat{\theta}_{MLE} &= \operatorname*{argmax}_{\theta} \,P(D; \theta) \\ &= \operatorname*{argmax}_{\theta} \begin{pmatrix} n_H + n_T \\ n_H \end{pmatrix} \theta^{n_H} (1 - \theta)^{n_T} \\ &= \operatorname*{argmax}_{\theta} \,\log\begin{pmatrix} n_H + n_T \\ n_H \end{pmatrix} + n_H \cdot \log(\theta) + n_T \cdot \log(1 - \theta) \\ &= \operatorname*{argmax}_{\theta} \, n_H \cdot \log(\theta) + n_T \cdot \log(1 - \theta) \end{align}
$$
Solve for $\theta$ by taking derivative and equating with zero
$$
\begin{align} \frac{n_H}{\theta} = \frac{n_T}{1 - \theta} \Longrightarrow n_H - n_H\theta = n_T\theta \Longrightarrow \theta = \frac{n_H}{n_H + n_T} \end{align}
$$
Coin Toss with Prior Knowledge: assume you have hunch that $\theta$ is close to 0.5, but sample size is small, so you don't trust your estimate. -> Add m imaginery throws that would result in $\theta$ = 0.5
$$
\hat{\theta} = \frac{n_H + m}{n_H + n_T + 2m}
$$
## Bayesian Way:
Model $\theta$ as random variable, drawn from distribution P($\theta$) ($\theta$ is NOT random variable associated with event in sample space, in frequentist statistics, this is forbidden. In Bayesian statistics, this is allowed and you can specify prior belief P($\theta$) defining what values you believe $\theta$ is likely to take on)
$$
P(\theta \mid D) = \frac{P(D\mid \theta) P(\theta)}{P(D)}
$$
- P($\theta$) is prior distribution over parameter(s) $\theta$ 
- P(D $\mid \theta$ ) is likelihood of data given parameter(s) $\theta$
- P($\theta \mid$ D) is posterior distribution over parameter(s) $\theta$ after we observed the data

Nature choice for prior P($\theta$) is Beta distribution:
$$
\begin{align} P(\theta) = \frac{\theta^{\alpha - 1}(1 - \theta)^{\beta - 1}}{B(\alpha, \beta)} \end{align}
$$
where $B(\alpha, \beta) = \frac{\Gamma(\alpha) \Gamma(\beta)}{\Gamma(\alpha+\beta)}$  is the normalization constant. Why is Beta distribution a good fit?
- models probabilities ($\theta$ lives on $\left[0, 1\right]$)
- it is of same distributional family as binomial distribution (conjugate prior) $\rightarrow$ math will turn out nicely
$$
\begin{align} P(\theta \mid D) \propto P(D \mid \theta) P(\theta) \propto \theta^{n_H + \alpha -1} (1 - \theta)^{n_T + \beta -1} \end{align}
$$

## Maximum a Posteriori Probability Estimation: 
We can choose $\hat{\theta}$ to be most likely $\theta$ given data. 

MAP Principle: Find $\hat{\theta}$ that maximized the posterior distribution P($\theta\mid$ D):
$$
\begin{align} \hat{\theta}_{MAP} &= \operatorname*{argmax}_{\theta} \,P(\theta \mid D) \\ &= \operatorname*{argmax}_{\theta} \, \log P(D \mid \theta) + \log P(\theta) \end{align}
$$
For coin flipping example, it would be:
$$
\begin{align} \hat{\theta}_{MAP} &= \operatorname*{argmax}_{\theta} \;P(\theta | Data) \\ &= \operatorname*{argmax}_{\theta} \; \frac{P(Data | \theta)P(\theta)}{P(Data)} && \text{(By Bayes rule)} \\ &= \operatorname*{argmax}_{\theta} \;\log(P(Data | \theta)) + \log(P(\theta)) \\ &= \operatorname*{argmax}_{\theta} \;n_H \cdot \log(\theta) + n_T \cdot \log(1 - \theta) + (\alpha - 1)\cdot \log(\theta) + (\beta - 1) \cdot \log(1 - \theta) \\ &= \operatorname*{argmax}_{\theta} \;(n_H + \alpha - 1) \cdot \log(\theta) + (n_T + \beta - 1) \cdot \log(1 - \theta) \\ &\Longrightarrow \hat{\theta}_{MAP} = \frac{n_H + \alpha - 1}{n_H + n_T + \beta + \alpha - 2} \end{align}
$$
- MAP is identical to MLE with $\alpha-1$ hallucinated heads and $\beta-1$ hallucinated tails
- As n $\rightarrow \infty, \hat\theta_{MAP} \rightarrow \hat\theta_{MLE}$ as $\alpha-1$ and $\beta-1$ become irrelevant compared to very large n_H, n_T

## "True" Bayesian Approach:
Use posterior predictive distribution directly to make prediction about label Y of test sample with features X:
$$
P(Y\mid D,X) = \int_{\theta}P(Y,\theta \mid D,X) d\theta = \int_{\theta} P(Y \mid \theta, D,X) P(\theta | D) d\theta
$$
Above is generally intractable in closed form.
For coin toss example, to make predictions using $\theta$, we can use:
$$
\begin{align} P(heads \mid D) =& \int_{\theta} P(heads, \theta \mid D) d\theta\\ =& \int_{\theta} P(heads \mid \theta, D) P(\theta \mid D) d\theta \ \ \ \ \ \ \textrm{(Chain rule: $P(A,B|C)=P(A|B,C)P(B|C)$.)}\\ =& \int_{\theta} \theta P(\theta \mid D) d\theta\\ =&E\left[\theta|D\right]\\ =&\frac{n_H + \alpha}{n_H + \alpha + n_T + \beta} \end{align}
$$
Here, we useed fact that we defined P(heads $\mid$ D, $\theta$) = P(heads $\mid \theta$) (this is only the case because we assumed that our data is drawn from binomial distribution - in general this would not hold)