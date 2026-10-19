# Theoretical Analysis

This document formalizes two claims that motivate the experimental design of EEL. Section 2 analyzes *how* the negative-sampling distribution determines whether contrastive training moves the encoder beyond pretrained semantic similarity; it makes a falsifiable prediction that is tested in Section V-C of the paper. Section 3 analyzes *what* the leave-one-view-out ablations of Section V-B identify, and shows that the ranking metrics reported in Table VI witness view contributions in only one direction.

Both results are analytical statements about idealized quantities. Neither is required for the empirical claims of the paper; they are included to make the estimands precise and to state the conditions under which the reported gaps admit an information-theoretic reading.

**Contents**

1. [Setup](#1-setup)
2. [Same-instance hard negatives remove the semantic shortcut](#2-same-instance-hard-negatives-remove-the-semantic-shortcut)
3. [View contributions and what leave-one-out ablations identify](#3-view-contributions-and-what-leave-one-out-ablations-identify)

**Notation.** $\mathbb{E}$ denotes expectation, $\Pr$ probability, $\sigma$ the logistic function, $\mathrm{H}(\cdot \mid \cdot)$ conditional Shannon entropy in nats, $I(\cdot\,;\cdot \mid \cdot)$ conditional mutual information, and $A \perp\!\!\!\perp B \mid C$ conditional independence. The candidate hunk is written $h$ (a value) or $H$ (a random variable); the entropy symbol $\mathrm{H}$ is upright.

---

## 1. Setup

Let $\mathcal{X}$ denote the instance space and $\mathcal{H}$ the space of candidate hunks. An instance $x = (q, t, o) \in \mathcal{X}$ consists of a requirement $q$, a test view $t$, and an original-source view $o$. *Maintainer retention* is the binary relation

```math
y^\ast : \mathcal{X} \times \mathcal{H} \to \lbrace 0, 1\rbrace,
```

where $y^\ast(x, h) = 1$ indicates that $h$ corresponds to the reference patch for instance $x$. As discussed in Section I of the paper, this is an operational approximation: retained hunks form $T_1(x) \cup T_2(x)$ and non-retained hunks form $T_3(x)$.

**Encoder and scoring.** Let $\phi_\theta$ denote the shared encoder producing $\ell_2$-normalized projections, with pairwise score

```math
f_\theta(u, v) = \frac{\phi_\theta(u)^\top \phi_\theta(v)}{\tau}, \qquad \tau > 0 .
```

Write $\phi_0 := \phi_{\theta_0}$ for the task-untrained backbone (model M0) and define the *semantic similarity*

```math
s_{\mathrm{sem}}(q, h) := \frac{\phi_0(q)^\top \phi_0(h)}{\tau},
```

the score available without task-specific training.

**Contrastive objective.** For an anchor $x$, a positive $h^+ \in T_1(x) \cup T_2(x)$, and negatives $h_1^-, \dots, h_K^-$, the InfoNCE term in the REQ→HUNK direction is

```math
\mathcal{L}(x; \theta) = -\log \frac{\exp\big(f_\theta(q, h^+)\big)}{\exp\big(f_\theta(q, h^+)\big) + \sum_{k=1}^{K} \exp\big(f_\theta(q, h_k^-)\big)} . \qquad (1)
```

We contrast two negative-sampling distributions:

- *Cross-repository random negatives* $h_k^- \sim p_{\mathrm{marg}}(h)$, drawn independently of $x$ (model M1).
- *Same-instance hard negatives* $h_k^- \sim p(h \mid x,\, y^\ast = 0)$, i.e., hunks from $T_3(x)$ (model M2).

---

## 2. Same-instance hard negatives remove the semantic shortcut

> **Assumption 1 (Semantic near-invariance of $T_3$).** There exists $\delta > 0$ such that for every instance $x$ with $T_3(x) \neq \emptyset$,
>
> ```math
> \Big| \mathbb{E}_{h^+ \sim T_1 \cup T_2(x)}\big[s_{\mathrm{sem}}(q, h^+)\big] - \mathbb{E}_{h^- \sim T_3(x)}\big[s_{\mathrm{sem}}(q, h^-)\big] \Big| < \delta ,
> ```
>
> and $\delta$ is small relative to the cross-instance semantic gap
>
> ```math
> \Delta_{\mathrm{sem}}^{\mathrm{marg}} := \Big| \mathbb{E}_{h^+ \sim T_1 \cup T_2(x)}\big[s_{\mathrm{sem}}(q, h^+)\big] - \mathbb{E}_{h^- \sim p_{\mathrm{marg}}}\big[s_{\mathrm{sem}}(q, h^-)\big] \Big| , \qquad \delta \ll \Delta_{\mathrm{sem}}^{\mathrm{marg}} .
> ```

Assumption 1 follows from how $T_3$ is constructed: non-retained hunks are produced by the same generator for the same requirement, and are therefore semantically close to retained hunks by design. The mechanism diagnostics of the paper (Section IV-D) measure this quantity directly: the mean M0 requirement similarity of $T_3$ hunks has a median of 0.492 across the 82 evaluation issues.

> **Assumption 2 (Existence of a residual signal).** There exists a feature map $\psi : \mathcal{X} \times \mathcal{H} \to \mathbb{R}^d$ orthogonal to the pretrained representation, $\mathbb{E}\big[\psi\, \phi_0^\top\big] = 0$, such that retention is not determined by semantic similarity alone,
>
> ```math
> \mathbb{E}\big[y^\ast(x, h) \,\big|\, s_{\mathrm{sem}}(q, h) = s\big] \not\equiv \mathrm{const} ,
> ```
>
> and the residual dependence is carried by $\psi$: for each instance $x$,
>
> ```math
> \gamma(x) := \mathbb{E}_{h^+ \sim T_1 \cup T_2(x)}\big[\psi(x, h^+)\big] - \mathbb{E}_{h^- \sim T_3(x)}\big[\psi(x, h^-)\big] \neq 0 .
> ```

Assumption 2 is empirically supported in two ways. First, if retention were determined by $s_{\mathrm{sem}}$, the task-untrained model M0 would already separate the tiers, contradicting its Perfect Separation Rate of 0.280 (Table VII). Second, the original-source alignment analysis (Table VIII(b)) identifies a concrete instantiation of $\psi$: under M2 the retained and $T_3$ groups separate in ORIG similarity with rank-biserial effect size $r = 0.978$, whereas under M1 the same measurement gives $r = 0.256$.

> **Theorem 1 (Gradient decomposition under negative sampling).** Consider the gradient of (1) at the pretrained initialization $\theta_0$, decomposed as $\nabla_\theta \mathcal{L} = g_{\mathrm{sem}} + g_\perp$, where $g_{\mathrm{sem}}$ lies along the parameter direction that scales the pretrained similarity $s_{\mathrm{sem}}$ and $g_\perp$ lies along the direction aligned with $\psi$. Under Assumptions 1 and 2:
>
> **(a) Random negatives.** If $h_k^- \sim p_{\mathrm{marg}}$, then $\lVert g_{\mathrm{sem}} \rVert = \Theta\big(\Delta_{\mathrm{sem}}^{\mathrm{marg}}\big)$ while $\mathbb{E}[g_\perp] = 0$ with variance vanishing in $K$. The expected update direction lies in the span of the pretrained representation.
>
> **(b) Same-instance hard negatives.** If $h_k^- \sim p(h \mid x, y^\ast = 0)$, then $\lVert g_{\mathrm{sem}} \rVert = O(\delta)$, while $\lVert g_\perp \rVert \geq c_0 \lVert \gamma(x) \rVert$ for a constant $c_0 > 0$ independent of $\delta$. The gradient is dominated by the direction orthogonal to the pretrained representation.

**Proof.** Differentiating (1) and using that the softmax weights satisfy $w^+ + \sum_k w_k = 1$, the gradient takes the form

```math
\nabla_\theta \mathcal{L} = \sum_{k=1}^{K} w_k \Big[ \nabla_\theta f_\theta(q, h_k^-) - \nabla_\theta f_\theta(q, h^+) \Big], \qquad w_k = \frac{\exp\big(f_\theta(q, h_k^-)\big)}{\exp\big(f_\theta(q, h^+)\big) + \sum_j \exp\big(f_\theta(q, h_j^-)\big)} . \qquad (2)
```

At $\theta_0$, decompose the per-example score gradient into the two tangent directions,

```math
\nabla_\theta f_\theta(q, h)\big|_{\theta_0} = \alpha(q, h)\, v_{\mathrm{sem}} + \beta(q, h)\, v_\perp(q, h), \qquad (3)
```

where $v_{\mathrm{sem}}$ increases the pretrained similarity and $v_\perp$ increases alignment with $\psi$. Substituting (3) into (2) gives

```math
\nabla_\theta \mathcal{L} = \underbrace{\sum_k w_k \big(\alpha(q, h_k^-) - \alpha(q, h^+)\big)\, v_{\mathrm{sem}}}_{g_{\mathrm{sem}}} + \underbrace{\sum_k w_k \big(\beta(q, h_k^-)\, v_\perp(q, h_k^-) - \beta(q, h^+)\, v_\perp(q, h^+)\big)}_{g_\perp} .
```

*Case (a).* Since $h_k^-$ is drawn independently of $x$ from $p_{\mathrm{marg}}$, the semantic coefficient satisfies $\mathbb{E}[\alpha(q, h_k^-)] - \alpha(q, h^+) = -\Theta\big(\Delta_{\mathrm{sem}}^{\mathrm{marg}}\big)$ by Assumption 1, so $\lVert g_{\mathrm{sem}} \rVert = \Theta\big(\Delta_{\mathrm{sem}}^{\mathrm{marg}}\big) \lVert v_{\mathrm{sem}} \rVert$. For the orthogonal component, independence of $h_k^-$ from $x$ implies that $`\mathbb{E}_{h^-}\big[\beta(q, h^-)\, v_\perp(q, h^-)\big]`$ is an $x$-independent baseline; its difference from the positive term has mean zero over the data distribution and is not systematically aligned with $\psi(x, \cdot)$, with variance decreasing in $K$ by averaging. Hence, the expected update lies along $v_{\mathrm{sem}}$.

*Case (b).* With $h_k^- \in T_3(x)$, Assumption 1 gives $|\alpha(q, h_k^-) - \alpha(q, h^+)| = O(\delta)$ and therefore $\lVert g_{\mathrm{sem}} \rVert = O(\delta) \lVert v_{\mathrm{sem}} \rVert$. For the orthogonal component, Assumption 2 gives a systematic within-instance separation in $\psi$, so

```math
g_\perp = -\,\bar{\beta}\,\gamma(x) + O(\delta), \qquad \bar{\beta} := \sum_k w_k\, \beta_k > 0 ,
```

which yields $\lVert g_\perp \rVert \geq c_0 \lVert \gamma(x) \rVert$ for some $c_0 > 0$ independent of $\delta$. Since $\lVert g_{\mathrm{sem}} \rVert = O(\delta)$ while $\lVert g_\perp \rVert = \Omega(1)$, the gradient is dominated by the direction orthogonal to the pretrained representation. $\blacksquare$

**Remark 1 (Falsifiable prediction).** Theorem 1(b) bounds the semantic component by $O(\delta)$ and the orthogonal component from below independently of $\delta$. The advantage of same-instance negatives should therefore *increase* as $\delta$ decreases, that is, as $T_3$ hunks approach retained hunks in requirement similarity. This prediction is tested by the stratified evaluation in Section V-C of the paper: Table VIII(a) splits the evaluation issues at the median M0 requirement similarity and finds that the M2 advantage over M1 grows from +0.294 to +0.436 nDCG@k as $\delta$ decreases. An explanation based on improved requirement matching alone predicts the opposite ordering.

**Remark 2 (Scope).** Theorem 1 concerns the update direction at initialization. It does not guarantee convergence to an optimal representation, and it does not order intermediate regimes such as same-repository cross-issue negatives (M3), which satisfy Assumption 1 only partially and empirically fall between M1 and M2 (Table VII). The decomposition also treats the REQ→HUNK direction in isolation; the full objective couples four pair types (Section III-B of the paper), and same-instance negatives are added in both the REQ→HUNK and ORIG→HUNK directions.

---

## 3. View contributions and what leave-one-out ablations identify

Let $(X, H, Y^\ast)$ be jointly distributed according to the data-generating process, with $X = (X_Q, X_T, X_O)$ decomposing into the requirement, test, and original-source views. For $i \in \lbrace Q, T, O\rbrace$ write $S_{-i} := \lbrace Q, T, O\rbrace \setminus \lbrace i\rbrace$, and let $X_S$ denote the components of $X$ indexed by $S$.

For the logistic loss $\ell(f, y) = -y \log \sigma(f) - (1 - y) \log(1 - \sigma(f))$ and a view set $S$, define the restricted Bayes risk

```math
\mathcal{R}^\ast(S) := \inf_{f \text{ measurable}} \mathbb{E}\big[\ell\big(f(X_S, H),\, Y^\ast\big)\big] .
```

> **Definition 1 (Conditional view contribution).** The *contribution* of view $i$ is the conditional mutual information
>
> ```math
> \Delta_i := I\big(Y^\ast ; X_i \,\big|\, X_{S_{-i}},\, H\big) .
> ```
>
> View $i$ is *conditionally redundant* if $\Delta_i = 0$, and the view set is *strictly complementary* if $\Delta_i > 0$ for every $i \in \lbrace Q, T, O\rbrace$.

All information measures condition on the candidate hunk $H$. Without this conditioning, a view could appear informative merely through correlation with properties of $H$ that are already available at scoring time; conditioning on $H$ isolates the *contextual* contribution of each view.

> **Theorem 2 (Ablation identifiability).** Under the logistic loss, for each $i \in \lbrace Q, T, O\rbrace$,
>
> ```math
> \mathcal{R}^\ast(S_{-i}) - \mathcal{R}^\ast(\lbrace Q, T, O\rbrace) = \Delta_i . \qquad (4)
> ```
>
> Consequently:
>
> **(a)** The view set is strictly complementary if and only if every leave-one-view-out model is strictly suboptimal, i.e., $\mathcal{R}^\ast(S_{-i}) > \mathcal{R}^\ast(\lbrace Q, T, O\rbrace)$ for all $i$.
>
> **(b) No single view suffices.** Under strict complementarity, for every singleton $\lbrace i\rbrace$,
>
> ```math
> \mathcal{R}^\ast(\lbrace i\rbrace) - \mathcal{R}^\ast(\lbrace Q, T, O\rbrace) \geq \max_{j \neq i} \Delta_j > 0 .
> ```
>
> **(c)** The empirical leave-one-view-out gap in validation cross-entropy estimates $\Delta_i$ up to the difference of the two models' excess risks.

**Proof.** *Step 1.* For the logistic loss, the Bayes-optimal scorer given inputs $Z$ is the log-odds $f^\ast(z) = \log \frac{\eta(z)}{1 - \eta(z)}$ with $\eta(z) = \Pr(Y^\ast = 1 \mid Z = z)$, and the minimal risk equals the conditional Shannon entropy in nats,

```math
\inf_f \mathbb{E}\big[\ell(f(Z), Y^\ast)\big] = \mathbb{E}\big[-Y^\ast \log \eta(Z) - (1 - Y^\ast) \log(1 - \eta(Z))\big] = \mathrm{H}(Y^\ast \mid Z) .
```

Applying this with $Z = (X_{S_{-i}}, H)$ and $Z = (X_Q, X_T, X_O, H)$ gives $\mathcal{R}^\ast(S_{-i}) = \mathrm{H}(Y^\ast \mid X_{S_{-i}}, H)$ and $\mathcal{R}^\ast(\lbrace Q, T, O\rbrace) = \mathrm{H}(Y^\ast \mid X_Q, X_T, X_O, H)$.

*Step 2.* By the chain rule for conditional mutual information,

```math
\Delta_i = \mathrm{H}(Y^\ast \mid X_{S_{-i}}, H) - \mathrm{H}(Y^\ast \mid X_i, X_{S_{-i}}, H) ,
```

and since $X_i$ together with $X_{S_{-i}}$ is $(X_Q, X_T, X_O)$, substituting the identities of Step 1 yields (4).

*(a)* is immediate, since the risk gap is positive if and only if the conditional mutual information is positive.

*(b)* Fix $i$ and any $j \neq i$. Since $\lbrace i\rbrace \subseteq S_{-j}$, conditioning on additional variables cannot increase entropy:

```math
\mathcal{R}^\ast(\lbrace i\rbrace) = \mathrm{H}(Y^\ast \mid X_i, H) \geq \mathrm{H}(Y^\ast \mid X_{S_{-j}}, H) = \mathcal{R}^\ast(S_{-j}) .
```

Subtracting $\mathcal{R}^\ast(\lbrace Q, T, O\rbrace)$ from both sides and applying (4) for index $j$ gives the bound; positivity follows from strict complementarity, and taking the maximum over $j \neq i$ sharpens it.

*(c)* follows from (4): replacing each Bayes risk by the empirical risk of a trained model introduces the corresponding excess risk, so the estimate carries a bias bounded by the difference of the two excess risks. $\blacksquare$

> **Corollary 1 (Non-negativity and the noise floor).** $\Delta_i \geq 0$ for every $i$, with equality if and only if $Y^\ast \perp\!\!\!\perp X_i \mid (X_{S_{-i}}, H)$. Any negative empirical gap is therefore attributable entirely to estimation and optimization error, and the magnitude of negative gaps observed across seeds calibrates the resolution at which small positive gaps can be distinguished from zero.

**Proof.** Non-negativity and the equality condition are standard properties of conditional mutual information; the second statement follows by combining them with Theorem 2(c). $\blacksquare$

### 3.1 Ranking metrics witness view contributions in one direction

The gaps reported in Table VI of the paper are differences in nDCG@k, $T_1$-Recall, $T_2$-Recall, and PSR rather than in cross-entropy, so they do not estimate $\Delta_i$ directly. The following proposition states exactly what they do and do not establish.

> **Proposition 1 (Metric gaps as one-sided witnesses).** Let $M$ be any performance functional that depends on a scorer only through the ranking it induces within each instance, and let $M^\ast(S)$ denote its value at the Bayes-optimal scorer for view set $S$. Then:
>
> **(i)** If $\Delta_i = 0$, then $M^\ast(S_{-i}) = M^\ast(\lbrace Q, T, O\rbrace)$ for every such $M$. Equivalently, a nonzero gap in *any* ranking metric witnesses $\Delta_i > 0$.
>
> **(ii)** The converse fails: $\Delta_i > 0$ is compatible with $M^\ast(S_{-i}) = M^\ast(\lbrace Q, T, O\rbrace)$ for all such $M$ simultaneously.

**Proof.** *(i)* By Corollary 1, $\Delta_i = 0$ is equivalent to $Y^\ast \perp\!\!\!\perp X_i \mid (X_{S_{-i}}, H)$, hence $\eta(X_{S_{-i}}, H) = \eta(X_Q, X_T, X_O, H)$ almost surely. The two Bayes-optimal scorers are strictly increasing transforms of the same posterior and therefore induce identical within-instance rankings, so any ranking-determined $M$ takes the same value.

*(ii)* Suppose the posterior factorizes as $\eta(x, h) = \sigma\big(g(x_{S_{-i}}, h) + c(x_i)\big)$ with $c$ non-constant. Then $\Delta_i > 0$, since $X_i$ shifts the posterior. But $c(x_i)$ is constant across candidates $h$ within an instance, so the induced within-instance ranking is unchanged and every ranking-determined $M$ is unaffected. All metrics used in the paper are computed within each instance and macro-averaged, and are therefore of this form. $\blacksquare$

**Remark 3 (Estimation protocol).** Proposition 1 implies that ranking gaps can be used in only one direction: a gap establishes that a view contributes, whereas a gap indistinguishable from zero does not establish redundancy. Reporting validation cross-entropy for each ablation, in addition to the ranking metrics, would make the estimand of Theorem 2 directly observable. Because $\Delta_i$ is a difference of two trained models' risks, paired estimation is preferable: train the full and ablated models from the same seeds and report the paired difference with a bootstrap interval, which removes the seed-level variance common to both terms.
