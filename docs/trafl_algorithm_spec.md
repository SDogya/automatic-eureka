# TraFL: подробная спецификация алгоритма и протокола воспроизведения

Источник: [Beyond Mode-Seeking RL: Trajectory-Balance Post-Training for Diffusion Language Models](32236_Beyond_Mode_Seeking_RL_T.pdf), анонимная версия «Under review as a conference paper at ICLR 2027», 25 страниц. Ссылки «с. N» означают печатный номер страницы PDF. Прочитаны основной текст, Algorithm 1 и приложения A–K. Документ описывает сведения именно этой версии статьи; опубликованные впоследствии реализации и настройки других работ не подставляются.

Статусы сведений: **авторское** — явно дано в статье; **математическое следствие** — раскрытие данной формулы, не новая настройка; **логическая нотация спецификации** — обозначение для механического описания, не заявленный авторами layout; **GAP** — необходимая деталь отсутствует. Наличие GAP означает, что статья сама по себе не определяет единственную побитно воспроизводимую реализацию.

**Отклонение, разрешённое пользователем:** более слабая **Qwen 0.8B dLLM** вместо авторского LLaDA-8B-Instruct. Это diffusion-модель согласно уточнению пользователя; точный checkpoint ID, внутренние размерности и API не заданы. Изменяется экспериментальная база: численные результаты статьи не становятся ожидаемыми результатами другой модели. Остальные опубликованные авторские компоненты ниже сохраняются, включая LoRA rank128/alpha128, score и learned logZ; привязка target modules и prompt hidden states к пользовательскому checkpoint требует отсутствующих сведений. Источник авторской базы: §4.1, с. 5; источник замены: пользовательское уточнение.

## 1. Input/Output & State Definition

### 1.1. Входы, выходы и единицы индексации

**Входы обучения:** начальная masked-diffusion policy $p_{\theta_{\mathrm{init}}}$; фиксированная reference policy $p_{\mathrm{ref}}$; функция reward $r$; набор prompts $D$; $\beta,G,K,M$. Именно эти входы перечислены в Algorithm 1, App. A, с. 14. Контекст $x$ — prompt, $y$ — окончательное completion, а не промежуточная denoising state (§3, с. 3).

**Выходы:** обученная policy $p_\theta$ и normalization predictor $Z_\phi$. Для отдельного prompt predictor выдаёт один скаляр $\log Z_\phi(x)$; $Z_\phi(x)$ положителен при конечном вещественном log-значении. Формат сохранения весов и inference API статья не задаёт (Algorithm 1, с. 14; §4.1, с. 5).

Для текста протокола индексы токенов, rollout, mask replicate, prompt и update **1-indexed**: $i=1,\ldots,L$, $g=1,\ldots,G$, $k=1,\ldots,K$, $b=1,\ldots,B$, $s=1,\ldots,M$. Индекс denoising state идёт от $T_{\mathrm{steps}}$ к $0$, как в авторской $\tau=(z_T,\ldots,z_0)$. Переименование $T_{\mathrm{steps}}$ отличает число шагов от temperature $T_{\mathrm{temp}}$; в статье оба обозначены $T$ (§3, с. 3; §4.2, с. 6; §5.1, с. 7).

### 1.2. Матрица символов, типов и логических размерностей

Формы ниже нужны для однозначных индексов. Статья не фиксирует оси физических тензоров, padding, packed layout или distributed sharding. $B=|D_b|$, $P_b$ — число prompt tokens, $V$ — размер словаря, $d$ — ширина hidden state: это **логические обозначения спецификации**, их численные значения не заданы. Для переменных длин структуры являются семействами последовательностей, а не утверждением о конкретном padded tensor (Algorithm 1, с. 14; §4.1, с. 5).

| Символ | Тип / логическая форма | Значение и диапазон | Источник |
|---|---|---|---|
| $D,D_b$ | набор prompts, batch из $B$ элементов | $D_b\sim D$; механизм выборки не задан | App. A, с. 14 |
| $x_b$ | token sequence длины $P_b$ | prompt вместе с инструкцией / форматом | App. E, с. 21 |
| $\theta,\theta_{\mathrm{init}}$ | семейство policy parameters | обновляемая policy и начальная | Algorithm 1, с. 14 |
| reference parameters | семейство parameters | frozen на всём протяжении post-training | §3.2, с. 4 |
| $\phi$ | семейство MLP parameters | обучаемая normalization head | §4.1, с. 5 |
| $y_b^{(g)}$ | token sequence длины $L_{bg}$ | окончательное completion rollout $g$ | §3, с. 3; App. A |
| $z_t,\tau$ | token state; последовательность states | $z_T$ fully noised, $z_0=y$ fully denoised | §3, с. 3 |
| $\mathcal T(y)$ | множество trajectories | все пути, заканчивающиеся в $y$ | Proposition 1, с. 3 |
| $r_{bg},\bar r_b,\widetilde r_{bg}$ | скаляры; логические массивы $B\times G$, $B$, $B\times G$ | binary $r\in\{0,1\}$; $\bar r\in[0,1]$; centered $\widetilde r=r-\bar r$ | §4.1; Algorithm 1 |
| $\beta$ | scalar $>0$ | reward sharpness, значение отсутствует | Eq. (1), с. 4 |
| $Z(x),\log Z(x)$ | scalar positive; scalar real | normalization точной theoretical target | Eqs. (1)–(3), с. 4 |
| $a_b=\log Z_\phi(x_b)$ | scalar; массив $B$ | практический predictor; общий для $G$ rollout данного prompt | Algorithm 1; §4.1 |
| $H_b$ | логическая матрица $P_b\times d$ | prompt hidden states для mean pooling | §4.1, с. 5 |
| $h_b$ | вектор $d$ | mean-pooled, detached features для head | §4.1, с. 5 |
| $l,m,y_l,q_l$ | целое; множество позиций; token sequence; corruption distribution | формально $l\in\{1,\ldots,L\}$, $|m|=l$, mask без replacement | Eq. (4); App. C |
| $E(y)$, $A(y)$ | множества индексов | EOS positions и non-EOS positions: вспомогательная нотация для ограничения §4.1 | §4.1, с. 5 |
| $p_\theta(y_i\mid y_l,x)$ | scalar в $[0,1]$ | probability исходного target token в замаскированной позиции | Eq. (4), с. 5 |
| denoising probabilities | логическое семейство $L\times V$ | predictions masked tokens за один model forward; физический layout не задан | App. C, с. 15 |
| $u_{bgk},v_{bgk}$ | scalar; логические массивы $B\times G\times K$ | masked-token average для current / reference | Eq. (4); Algorithm 1 |
| $S_{\theta,bg},S_{\mathrm{ref},bg}$ | scalar; массивы $B\times G$ | Monte Carlo estimates $\log\widehat p$ | §3.2, с. 5 |
| $\delta_{bg}$ | scalar; массив $B\times G$ | practical balance residual | Algorithm 1 |
| $\mathcal L_{\mathrm{TraFL}}$ | scalar $\geq0$ при конечных residuals | batch mean squared residual | Algorithm 1 |
| $\mathcal L_\theta(y\mid x)$ | scalar | masked-diffusion ELBO; не training loss $\mathcal L_{\mathrm{TraFL}}$ | Eq. (6), с. 15 |
| $\sigma,p_\theta^\sigma,\bar p_\theta$ | permutation; distribution; distribution | uniform reveal order и его mixture | App. C, с. 15 |
| $P_F,P_B$ | trajectory distribution; conditional backward distribution | conceptual exact TB, не отдельные practical networks | App. D.4, с. 19 |
| $p_i,\varphi_i,q_i,A_y,\bar r$ | scalars / vectors over paths | tabular theory: $p_i\propto e^{\varphi_i}$, $q_i=p_i/p_y$, $A_y=r(x,y)-\bar r$ | App. D.2, с. 17 |
| $J,H,\operatorname{supp},g$ | scalar objective; entropy; support; deterministic map | theoretical properties, не дополнительные losses | §3.1; App. D.1 |
| $\Omega_T,\Omega_M,\mathsf T,\mathsf M$ | countable sets; random variables | trajectories, terminal modes; переименование random $M$ отличает его от update count | Theorem 1, с. 4 |
| $o^{(n)},p_v,\ell^{(n)},w^{(n)},\alpha_{\mathrm{DMPO}},L_{\mathrm{DCE}}$ | rollout; sampling distribution; scalar score / weight; positive scale; denoising CE | символы анализа DMPO, не параметры TraFL; $\alpha_{\mathrm{DMPO}}$ отличается от LoRA $\alpha$ | App. D.6, с. 20–21 |
| $n,c,k$ в Pass@$k$ | целые | $0\leq c\leq n$, $1\leq k\leq n$; evaluation sample budget | §4.2, с. 6 |
| $S_t^i,\bar J$ | set token indices; scalar | unmasked positions sample $i$ на step $t$ и усреднённый Jaccard | §5.1, с. 7 |
| $D_A,D_B,S_{\mathrm{LLM}}$ | sets of responses; categorical function | judge comparison, winner $A/B/\mathrm{Tie}$ | App. H, с. 23 |

$K$ в App. D.3 обозначает количество теоретических trajectories; $K=4$ в Algorithm 1 — количество mask replicates. Эти $K$ не отождествляются. Аналогично $\varphi$ табличной softmax theory не является $\phi$ normalization MLP. Совпадение авторских букв не означает общие параметры (App. A; App. D.2–D.3).

### 1.3. Авторские константы и настройки

| Компонент | Авторское значение / правило | Источник |
|---|---|---|
| Base | LLaDA-8B-Instruct, instruction-tuned masked diffusion; TraFL непосредственно, без дополнительного SFT | §4.1, с. 5 |
| Fine-tuning | LoRA rank $r=128$, scaling factor $\alpha=128$ | §4.1, с. 5 |
| Reference | frozen reference model | §3.2, с. 4 |
| Normalization | $\log Z(x)$: 2-layer MLP над mean-pooled prompt hidden states; backbone features detached | §4.1, с. 5 |
| Peak LR | $6\times10^{-5}$ | §4.1, с. 5 |
| LR schedule | cosine decay, minimum $6\times10^{-6}$ | §4.1, с. 5 |
| Rollouts | $G=5$ per prompt | §4.1, с. 5 |
| Score replicates | $K=4$ antithetic replicates, complementary mask pairs | §4.1, с. 5 |
| Masked domain | only non-EOS positions; same masking pattern for current / reference | §4.1; Eq. (4), с. 5 |
| Online updates | fresh current-policy rollouts; rewards / scores на том же batch; exactly one gradient update per step | App. A, с. 14 |
| Reward | binary: exact answer matching в math, execution-based test passing в code | §4.1, с. 6 |
| Evaluation max completion | 256 и 512 tokens | §4.2, с. 6 |
| Evaluation block | 32 tokens | §4.2, с. 6 |
| Evaluation sampler | fastdLLM parallel decoding, low-confidence masking, stochastic nucleus | §4.2, с. 6 |
| Evaluation top-$p$ | 0.91 | §4.2, с. 6 |
| Main evaluation temperature | $T_{\mathrm{temp}}=0.6$ | §5.2–5.4; Figs. 1–3 |
| Temperature sweep | $\{0.3,0.6,0.9\}$ | §5.2, с. 7 |
| Evaluation samples | $n=16$ independent generations per problem for reported Pass@$k$ | Fig. 1, с. 7; Fig. 2, с. 8; App. B |
| Budgets | $k\in\{1,2,4,8,16\}$; отдельно Pass@5 | §5.2–5.3, с. 7–8 |
| Path-diversity protocol | $N=16$, $L=256$ generated tokens, $T_{\mathrm{steps}}=256$ | §5.1, с. 7 |
| Judge | GPT-5-4, max completion 256, temperature 0.6, response sets of 16, randomized set order, blind to method identity | §5.5; App. H; AI use statement, с. 10 |
| Significance threshold | prespecified one-sided exact sign tests on non-ties; $p\leq0.05$ | App. H, с. 23–24 |
| Compute | 8 modern accelerator cards (NPU/TPU/GPU), at least 80 GB memory; math run approximately 10 hours, code approximately 18 hours | App. J, с. 25 |

Значения $\beta,M,B$ не опубликованы. Длительности App. J не являются значениями $M$. Evaluation lengths / temperature / block / top-$p$ не объявлены тренировочными rollout settings (§4.1–4.2; App. A, J).

### 1.4. Теоретическая цель, которую нельзя подменять практическим loss

Для raw terminal reward и фиксированного $x$ точная reference-anchored target:

$$p^*(\tau\mid x)=Z(x)^{-1}p_{\mathrm{ref}}(\tau\mid x)e^{\beta r(x,y)}.\tag{1}$$

$$\delta(\tau,x)=\log p_\theta(\tau\mid x)-\log p_{\mathrm{ref}}(\tau\mid x)-\beta r(x,y)+\log Z(x).\tag{2}$$

$$\mathcal L_{\mathrm{TraFL}}=\mathbb E_{\tau\sim p_\theta(\cdot\mid x)}[\delta(\tau,x)^2].\tag{3}$$

Eqs. (1)–(3), §3.2, с. 4. По нормировке (**математическое следствие**, App. D.5, с. 20):

$$Z(x)=\sum_\tau p_{\mathrm{ref}}(\tau\mid x)e^{\beta r(x,y(\tau))},\qquad p^*(y\mid x)=\frac{p_{\mathrm{ref}}(y\mid x)e^{\beta r(x,y)}}{Z(x)},\qquad p^*(\tau\mid y,x)=p_{\mathrm{ref}}(\tau\mid y,x).$$

Это утверждения о точной target; они не гарантированы для neural policy после минимизации практического surrogate. Для binary raw reward $1\leq Z(x)\leq e^\beta$ — следствие Eq. (1), **не** ограничение или clamp на обучаемый $\log Z_\phi$.

Разложение точного residual (**математическое раскрытие** §3.2, с. 5; App. D.5, с. 20):

$$\delta(\tau,x)=\underbrace{\log p_\theta(y\mid x)-\log p_{\mathrm{ref}}(y\mid x)-\beta r(x,y)+\log Z(x)}_{\text{completion term}}+\underbrace{\log p_\theta(\tau\mid y,x)-\log p_{\mathrm{ref}}(\tau\mid y,x)}_{\text{within-completion path term}}.$$

Практический loss использует completion-level masked score. Второй член не вычисляется отдельно, trajectories не превращаются в дополнительный loss, отдельная backward model не обучается (§3.2; App. D.5).

### 1.5. Что именно оценивает masked-denoising score

Формальная Eq. (4), §3.2, с. 5, при $L\geq1$:

$$\log\widehat p_\theta(y\mid x)=\mathbb E_{l\sim\mathrm{Uniform}\{1,\ldots,L\}}\mathbb E_{y_l\sim q_l(\cdot\mid y,x)}\left[\frac1l\sum_{i=1}^L\mathbf1[y_i^l=\mathrm{[mask]}]\log p_\theta(y_i\mid y_l,x)\right].\tag{4}$$

$q_l$ выбирает **ровно $l$** uniformly random positions без replacement. Вероятность каждого consistent corruption равна $\binom{L}{l}^{-1}$, остальных — 0. Predict all masked positions in one forward pass; joint reconstruction factorization $p_\theta(y\mid y_l,x)=\prod_{i:y_i^l=\mathrm{[mask]}}p_\theta(y_i\mid y_l,x)$ (App. C, с. 15). В likelihood берётся **исходный** token $y_i$, не mask token и не заново sampled prediction.

One-step reconstruction marginal $p_\theta^{(l)}(y\mid x)=\mathbb E_{y_l\sim q_l(\cdot\mid y)}[p_\theta(y\mid y_l,x)]$ и Jensen bound:

$$\log p_\theta^{(l)}(y\mid x)\geq\mathbb E_{y_l\sim q_l(\cdot\mid y)}\left[\sum_{i=1}^L\mathbf1[y_i^l=\mathrm{[mask]}]\log p_\theta(y_i\mid y_l,x)\right].\tag{5}$$

$$\log\widehat p_\theta(y\mid x)=\frac1L\mathcal L_\theta(y\mid x),\quad \mathcal L_\theta(y\mid x)=\mathbb E_l\mathbb E_{y_l}\left[\frac L l\sum_{i=1}^L\mathbf1[y_i^l=\mathrm{[mask]}]\log p_\theta(y_i\mid y_l,x)\right].\tag{6}$$

App. C, с. 15, Eqs. (5)–(6). Пусть $\sigma$ — uniform reveal-order permutation и $p_\theta^\sigma(y\mid x)=\prod_{j=1}^L p_\theta(y_{\sigma_j}\mid y_{\sigma_{<j}},x)$, где unrevealed positions остаются masked. Тогда:

$$\mathcal L_\theta(y\mid x)=\mathbb E_\sigma[\log p_\theta^\sigma(y\mid x)]\leq\log\bar p_\theta(y\mid x),\quad\bar p_\theta(y\mid x)=\mathbb E_\sigma[p_\theta^\sigma(y\mid x)].$$

Score — expected reveal-order log probability, **делённая на completion length**. Jensen equality только при invariant likelihood across reveal orders. Monte Carlo unbiased для этого expected score; в общем случае не unbiased, не consistent и не order-preserving estimator exact completion log probability под используемым low-confidence sampler. Approximation bias policy–reference difference может иметь любой знак и не обязан исчезать. Общие masks уменьшают variance разности, но не устраняют bias (App. C, с. 15–16).

Нельзя заменять $1/l$ на сумму без нормировки, $1/L$, multiplication by $L$, exponentiated probability или tokenwise PPO ratios. Eq. (6) даёт отношение к ELBO, но Algorithm 1 подставляет **Eq. (4)**, не ненормированную $\mathcal L_\theta$ (Algorithm 1, с. 14).

### 1.6. Почему path diversity обсуждается, и границы теоретических утверждений

Terminal-reward objective $J(x)=\sum_y p_\theta(y\mid x)r(x,y)$ с $p_\theta(y\mid x)=\sum_{\tau\in\mathcal T(y)}p_\theta(\tau\mid x)$ invariant к redistribution внутри $\mathcal T(y)$ при сохранении marginal. Это Proposition 1 (§3.1, с. 3; App. D.1, с. 16), а не gradient rule TraFL.

Theorem 1 для countable trajectory set и deterministic terminal-mode mapping $\mathsf M=g(\mathsf T)$: $|\operatorname{supp}(\mathsf M)|\leq|\operatorname{supp}(\mathsf T)|$, $H(\mathsf M)\leq H(\mathsf T)$. Доказательство: $\operatorname{supp}(\mathsf M)=g(\operatorname{supp}(\mathsf T))$, $H(\mathsf M\mid\mathsf T)=0$, $H(\mathsf T,\mathsf M)=H(\mathsf T)=H(\mathsf M)+H(\mathsf T\mid\mathsf M)$. Corollary: paths confined to $S$ покрывают не более $g(S)$; один path — один mode. Обратного достаточного условия нет: высокая path diversity сама по себе не доказывает diverse correct solutions (§3.1, с. 4; App. D.1, с. 16–17; §5.5, с. 9).

Proposition 2 касается **tabular softmax trajectories и exact Euclidean gradient ascent по logits**, не произвольной neural policy. При $p_i\propto e^{\varphi_i}$, $J=\sum_i p_i r_i$, $\bar r=\sum_i p_i r_i$, $p_y=\sum_{i\in\mathcal T(y)}p_i$, $q_i=p_i/p_y$, $A_y=r(x,y)-\bar r>0$:

$$\dot\varphi_i=p_i(r_i-\bar r),\quad\dot\varphi_i=p_y A_y q_i\quad(i\in\mathcal T(y)),$$
$$\frac{d}{dt}\log(q_i/q_j)=p_yA_y(q_i-q_j),\quad\dot q_i=p_y A_yq_i\big(q_i-\mathbb E_q[q]\big),$$
$$\frac d{dt}H(q)=-p_y A_y\operatorname{Cov}_q(q_i,\log q_i)\leq0.$$

Для $q_i>q_j$ ratio растёт; entropy строго падает при non-uniform $q$ на support. Это направление динамики, **не** доказательство asymptotic collapse без persistence assumptions. Shared neural parameters, KL, clipping, entropy bonuses и finite samples могут менять динамику. Sampling дополнительно усиливает concentration (App. D.2, с. 17).

App. D.3 — отдельная auxiliary theory для effective path scores $r_i$, которые могут различаться даже внутри одного completion:

$$J(q)=\sum_{i=1}^{K_{\mathrm{paths}}}q_i r_i.\tag{7}$$
$$q_i=\frac{e^{\varphi_i}}{\sum_j e^{\varphi_j}},\quad\frac{\partial J}{\partial\varphi_i}=q_i(r_i-\bar r),\quad\bar r=\sum_jq_jr_j.\tag{8}$$
$$\dot\varphi_i=r_i.\tag{9}$$
$$\dot q_i=q_i(r_i-\bar r).\tag{10}$$
$$q_i(t)=\frac{q_i(0)e^{r_i t}}{\sum_jq_j(0)e^{r_jt}}.\tag{11}$$

Eqs. (9)–(11) предполагают **additive payoff / mirror-descent / natural-gradient geometry**: ratio $q_i(t)/q_j(t)=[q_i(0)/q_j(0)]e^{(r_i-r_j)t}$, $\dot\varphi_i=\eta r_i$ лишь rescaling time. Обычный Euclidean ascent Eq. (8) вместо этого даёт $\dot q_i\propto q_i[q_i(r_i-\bar r)-\sum_jq_j^2(r_j-\bar r)]$ и не имеет Eq. (11) как решения. При equal scores additive dynamics сохраняет ratios и не объясняет within-completion locking pure terminal reward; эту ситуацию разбирает D.2. Linear objective maximizers: vertex при одном dominant score, face при ties. Эти формулы не задают optimizer TraFL (App. D.3, с. 18–19).

Conceptual exact TB с normalized backward distribution:

$$\log Z(x)+\log P_F(\tau\mid x)=\beta r(x,y)+\log P_B(\tau\mid y,x),\quad\forall\tau\in\mathcal T(y).\tag{12}$$
$$P_F(\tau\mid x)=Z(x)^{-1}e^{\beta r(x,y)}P_B(\tau\mid y,x).\tag{13}$$
$$P_F(y\mid x)=Z(x)^{-1}e^{\beta r(x,y)}.\tag{14}$$
$$P_F(\tau\mid y,x)=P_B(\tau\mid y,x).\tag{15}$$

App. D.4, с. 19. **Это концептуальные Eqs. (12)–(15), не practical TraFL**: Eq. (14) не содержит $p_{\mathrm{ref}}(y\mid x)$, в отличие от reference-anchored Eq. (1) / D.5. Нельзя подменять target Eq. (1) этим marginal или добавлять backward-network training.

Практический path invariance: при фиксированных completion $y$, prompt $x$, group context и independent corruption $m$ residual одинаков для двух realized paths $\tau,\tau'\in\mathcal T(y)$: $\widehat\delta(\tau,x;m)=\widehat\delta(\tau',x;m)=\widehat\delta(y,x;m)$. Ожидаемый сигнал не зависит от realized decoding order. Это устраняет direct path-dependent reinforcement; shared parameter updates всё равно могут менять $p_\theta(\tau\mid y,x)$ косвенно. Preservation reference-level diversity — эмпирический результат, не точная constraint surrogate (App. D.5, с. 20; §5.1, с. 7).

### 1.7. Критические точки соответствия

1. Frozen reference и current model сравниваются на **одинаковых independent corruption masks**, а не на realized rollout path (Eq. (4); App. D.5).
2. Score — masked-position **mean $1/l$**, затем Monte Carlo average (Eq. (4)); не exact likelihood и не ELBO sum.
3. В training reward center по **$G=5$ rollout одного prompt**, без деления на standard deviation (Algorithm 1).
4. $+\log Z_\phi(x)$, $-\beta\widetilde r$ и $S_\theta-S_{\mathrm{ref}}$ имеют именно эти знаки (Algorithm 1).
5. $\log Z$ — learned prompt-dependent MLP; feature detachment обязателен (§4.1).
6. Одна squared-loss update на fresh current-policy batch (App. A).
7. $K=4$ antithetic replicates, only non-EOS; точная конструкция complementary masks не опубликована (§4.1).
8. Raw theoretical reward и stochastic centered training reward не являются одной и той же exact target (App. D.5).
9. Evaluation fastdLLM settings не подставляются молча в training sampler (§4.1–4.2).
10. Более слабый checkpoint пользователя — явная смена базы; остальные настройки, отсутствующие в бумаге, не восстанавливаются догадками (§4.1; пользовательское задание).

## 2. Initialization Routine

### 2.1. Подготовить независимые training runs

**Inputs:** исходные data releases и нужная task family. **Operation:** создать **три** независимых запуска: GSM8K, MATH, code mixture. Math runs не объединяются в один checkpoint; code использует один combined checkpoint (§4.1, с. 5–6; Tab. 3, с. 23).

| Run | Training data | Число примеров | Evaluation mapping |
|---|---|---:|---|
| GSM8K | GSM8K training | 7,473 | GSM8K; Minerva Math как GSM8K-trained |
| MATH | MATH training | 7,500 | MATH-500; Minerva Math как MATH-trained |
| Code | concatenated filtered AceCode-89K + KodCode-Light-RL-10K | 11,937 + 2,695 = 14,632 | HumanEval, MBPP, LiveCodeBench |

Источники: §4.1–4.2, с. 5–6; Fig. 3, с. 9; App. F.1–F.2, с. 22–23. Dataset revision IDs, exact retained example IDs и prompt tokenization не опубликованы.

**AceCode filtering:** использовать existing CodeLlama-Instruct rollout metadata, поставляемые с AceCode-89K. Оставить problems с substantial reward uncertainty, убрать uniformly passed и uniformly failed; результат 11,937. Авторы не утверждают, что простое правило $0<\mathrm{pass\ rate}<1$ само по себе воспроизводит final subset: substantial uncertainty threshold отсутствует (App. F.1, с. 22).

**KodCode filtering:** из GPT-4o и DeepSeek-R1 rollout вычислить per-example pass-rate statistics; определить weaker proxy в среднем; отобрать high pass-sequence variance under that weaker proxy; затем lower-pass-rate portion этого subset; результат 2,695. Какая proxy оказалась weaker, variance formula, thresholds и retained fraction не заданы. Concatenate два filtered subsets; filtering мотивировано информативным binary feedback, а не stylistic/domain filtering (App. F.1, с. 22).

### 2.2. Создать policy, frozen reference и adapters

**Inputs:** $p_{\theta_{\mathrm{init}}},p_{\mathrm{ref}}$. **Operation:** присвоить $p_\theta\leftarrow p_{\theta_{\mathrm{init}}}$, использовать LoRA $r=128,\alpha=128$, без дополнительной SFT стадии. Reference заморожен. **State transformation:** создать отдельный обновляемый policy state и неизменяемое reference state; $\theta$ обозначает policy, реально обучаемые policy parameters задаёт LoRA. **Outputs:** current policy и frozen comparator (§4.1, с. 5; Algorithm 1, с. 14).

[EXPLICIT GAP IN PAPER: точный checkpoint revision, способ получения pref из init, список LoRA target modules, LoRA initialization/dropout/bias и дополнительные обучаемые policy parameters не указаны; авторский дефолт отсутствует.]

### 2.3. Создать normalization head

**Inputs:** prompt hidden states $H(x)$. **Operation:** mean-pool prompt states, detach features, подать в **2-layer MLP** с одним scalar output $a(x)=\log Z_\phi(x)$. Математическая запись вычислительного графа, с неуказанными внутренними MLP dimensions:

$$h(x)=\operatorname{MeanPool}_{\mathrm{prompt}}H(x),\qquad a(x)=\operatorname{MLP}_\phi(\operatorname{Detach}(h(x))).$$

**State transformation:** initialize $\phi$; initialize predictor $Z_\phi$ как в Algorithm 1. **Outputs:** $a(x)\in\mathbb R$. Head optimization не изменяет backbone prompt representations через этот вход; current-policy masked score остаётся trainable (§4.1, с. 5).

[EXPLICIT GAP IN PAPER: hidden-state layer и source backbone для head, учитываемые prompt special/padding tokens, скрытая ширина/activation/bias MLP, head initialization, отдельная LR/optimizer head, режим dropout и recomputation timing prompt features не указаны; авторский дефолт отсутствует.]

### 2.4. Зафиксировать prompt format

Одинаковый format используется для train и evaluation; все task instructions и response-format constraints находятся в **user prompt**, отдельного system prompt нет. Подписи boxes «System Prompt» не отменяют прямое пояснение App. E, с. 21. Примеры sampled program и math solution — иллюстрации, не данные для обязательного SFT. Их программный код здесь не копируется.

Math user instruction (App. E, с. 21, без исправления двух точек в конце):

> You are a math expert. You will be given a question to solve.
> Solve it step by step. Wrap the final answer in \boxed{} inside the &lt;answer&gt; tags..

Затем label **Response Format**, строка ` <reasoning> Your reasoning here </reasoning> ` и строка ` <answer> \boxed{...} </answer> `; далее task statement, показанный под label **[Task Prompt]**. Inline literal tags здесь — текст формата ответа, не программный код (App. E, с. 21–22).

Coding instruction (App. E, с. 21):

> You are a coding expert. You will be given a coding problem to solve.
> Solve it step by step. Ensure you wrap the answer in \`\`\`python\`\`\`.

Затем **Response Format**, строка ` <reasoning> Your reasoning here </reasoning> ` и строка **&lt;answer&gt; \`\`\`python ... \`\`\` &lt;/answer&gt;**; далее task statement под **[Task Prompt]**. Здесь передан literal fence marker, без блока кода. PDF типографически отображает grave characters неоднозначно; [EXPLICIT GAP IN PAPER: точная исходная prompt serialization, пробелы/переносы и tokenizer chat template не приложены; авторский дефолт отсутствует.]

### 2.5. Настроить update count и random state

**Inputs:** отсутствующие $M,B,\beta$ и optimizer configuration должны быть внешними явно документированными inputs, пока авторская информация не получена. **Operation:** подготовить cosine decay с заданными peak/min LR; начать step $s=1$; обучать до $s=M$. **State:** определить $\theta_1=\theta_{\mathrm{init}}$, $\phi_1=\phi_{\mathrm{initialized}}$ — параметры **до** первой update. Шаг $s$ читает $(\theta_s,\phi_s)$ и создаёт $(\theta_{s+1},\phi_{s+1})$; финальный результат после $M$ updates — $(\theta_{M+1},\phi_{M+1})$. Counters, random sampling state и optimizer state логически необходимы для стохастического Algorithm 1. Эти индексы — нотация спецификации. **Outputs:** инициализированный run (Algorithm 1, с. 14; §4.1, с. 5).

[EXPLICIT GAP IN PAPER: M, число эпох, batch size, random seeds, optimizer family/parameters, warmup, cosine horizon и scheduler step timing не указаны; авторский дефолт отсутствует.]

Это **детерминированное описание порядка действий при заданных входах и random draws**, не заявление о deterministic sampling или опубликованном seed.

## 3. Step-by-Step Execution Protocol (Deterministic Pseudocode)

### 3.1. Шаг S.1 — взять свежий prompt batch

**Inputs:** run dataset $D$, current $\theta_s$, step $s\in\{1,\ldots,M\}$.

**Mathematical Operation:** $D_b\sim D$, $B=|D_b|$.

**State Transformation:** создать batch-local context; policy/reference/head пока не обновлять.

**Outputs:** $x_1,\ldots,x_B$.

Источник: Algorithm 1, с. 14. [EXPLICIT GAP IN PAPER: B, replacement/shuffle/epoch policy, distributed batching и handling повторяющихся prompts не указаны; авторский дефолт отсутствует.]

### 3.2. Шаг S.2 — получить $G=5$ completion на prompt

**Inputs:** $x_b$, current policy $p_{\theta_s}$.

**Mathematical Operation:** для каждого $b$ получить $\{y_b^{(g)}\}_{g=1}^5\sim p_{\theta_s}(\cdot\mid x_b)$, путём masked diffusion generation.

**State Transformation:** сохранить окончательные completion и их token sequences в batch-local state. Внутри текущей update policy ещё не меняется; все rollout относятся к pre-update current policy.

**Outputs:** $B\times5$ completion с длинами $L_{bg}$.

Источник: Algorithm 1, App. A, с. 14; §4.1, с. 5. Realized trajectories не используются для masked score (App. D.5). [EXPLICIT GAP IN PAPER: training max length, denoising steps, block size, temperature, top-p, low-confidence thresholds, stopping/EOS rules, special tokens и точная независимость training rollout draws не заданы; авторский дефолт отсутствует.]

### 3.3. Шаг S.3 — вычислить raw binary rewards

**Inputs:** problem / reference answer или tests, $x_b$, $y_b^{(g)}$.

**Mathematical Operation:** $r_{bg}=r(x_b,y_b^{(g)})\in\{0,1\}$: math — exact answer matching; code — execution-based test passing. На evaluation math используется extracted boxed answer (§4.1–4.2, с. 6).

**State Transformation:** записать $r_{bg}$; reward не является обучаемой neural function в приведённом Algorithm 1.

**Outputs:** массив rewards $B\times5$.

[EXPLICIT GAP IN PAPER: training answer extraction, equivalence/canonicalization, обработка отсутствующей/нескольких boxed answers, malformed tags/fences, code extraction, tests aggregation, execution language/runtime/timeouts и reward для crash/невалидного completion не указаны; авторский дефолт отсутствует.]

### 3.4. Шаг S.4 — center reward строго внутри одного prompt group

**Inputs:** $r_{b1},\ldots,r_{b5}$.

**Mathematical Operation:**

$$\bar r_b=\frac15\sum_{g'=1}^{5}r_{bg'},\qquad\widetilde r_{bg}=r_{bg}-\bar r_b.$$

**State Transformation:** сохранить $\widetilde r_{bg}$; ни mean across prompt batch, ни division by standard deviation не выполнять в авторской формуле.

**Outputs:** centered rewards $B\times5$; $\sum_g\widetilde r_{bg}=0$ — **математическое следствие**.

Источник: Algorithm 1, с. 14. При $m$ correct rollout из 5 успешные имеют $1-m/5$, неуспешные $-m/5$. Если all success или all failure, $\widetilde r_{bg}=0$ для всей группы; статья не говорит её пропускать.

**Отличие от Eqs. (1)–(3):** batch group mean — random quantity, зависящая от sampled group, а не только от $x$. Поэтому его нельзя в общем случае точно поглотить deterministic $Z(x)$. Авторы называют centering stochastic variance-reduction heuristic, не exact rewriting target (App. D.5, с. 20).

### 3.5. Шаг S.5 — сэмплировать completion corruptions независимо от realized path

**Inputs:** окончательное $y_b^{(g)}$, $K=4$, non-EOS eligibility из §4.1.

**Mathematical Operation, формальная Eq. (4):** для каждого replicate выбрать $l\sim\mathrm{Uniform}\{1,\ldots,L\}$, затем множество $m\subset\{1,\ldots,L\}$ размера $l$ uniform без replacement; $y_i^l=\mathrm{[mask]}$ при $i\in m$, иначе $y_i^l=y_i$. Prompt $x$ — условие, completion — объект corruption.

**Авторская practical настройка:** использовать **4 antithetic replicates, complementary mask pairs**, mask только non-EOS positions. Random corruption не зависит от realized decoding path. Один и тот же pattern применяется к current и reference, включая одинаковые visible targets и masked positions (§4.1; Eq. (4), с. 5; App. D.5, с. 20).

**State Transformation:** сформировать four corrupted completion и sets masked positions на каждый rollout; сохранить pattern для обоих forward passes.

**Outputs:** $m_{bgk}$, $y_{bgk}^{\mathrm{masked}}$, $l_{bgk}=|m_{bgk}|$ для $k=1,\ldots,4$.

**Граница определённости:** $K=4$ означает **четыре mask replicates всего**, организованные в complementary pairs; внутри antithetic пары masks зависимы. Independence здесь означает независимость corruption от realized denoising path, не независимость complementary replicates друг от друга (§4.1, с. 5; Algorithm 1, с. 14; App. D.5, с. 20). Формальная Eq. (4) описывает $L$ completion positions, тогда как practical experiment исключает EOS; статья не выводит новое $q_l$ для eligible set. Нельзя молча задать $L=|A(y)|$, оставив иной normalization, или заявить, что EOS участвует в masked likelihood. [EXPLICIT GAP IN PAPER: как L в Eq. (4) связан с EOS-excluded length, как выбираются antithetic levels/sets, joint distribution пар, weighting, rounding и handling zero-sized complement не указаны; авторский дефолт отсутствует.]

### 3.6. Шаг S.6 — оценить current/reference score на одинаковых masks

**Inputs:** $x_b$, target $y_b^{(g)}$, четыре совместно заданных corruptions $m_{bgk}$, current $\theta_s$, frozen reference.

**Mathematical Operation:** для replicate с $l_{bgk}>0$ предсказать masked tokens обеими моделями и вычислить:

$$u_{bgk}=\frac1{l_{bgk}}\sum_{i\in m_{bgk}}\log p_{\theta_s}(y_{bg,i}\mid y_{bgk}^{\mathrm{masked}},x_b),$$
$$v_{bgk}=\frac1{l_{bgk}}\sum_{i\in m_{bgk}}\log p_{\mathrm{ref}}(y_{bg,i}\mid y_{bgk}^{\mathrm{masked}},x_b).$$

Для обычного equal-weight Monte Carlo evaluation Eq. (4), **математическая запись mean**, условная на корректно заданный sampling scheme:

$$S_{\theta,bg}=\frac14\sum_{k=1}^4u_{bgk},\qquad S_{\mathrm{ref},bg}=\frac14\sum_{k=1}^4v_{bgk}.$$

Это не доказывает unbiasedness неописанного antithetic implementation; детали coupling / EOS должны быть раскрыты отдельно (App. C, с. 16; §4.1, с. 5).

**State Transformation:** сохранить два score $B\times5$. В current score сохранить зависимость от trainable $\theta$; reference weights не обновлять. Общая corruption используется именно внутри разности $S_\theta-S_{\mathrm{ref}}$.

**Outputs:** $S_{\theta,bg},S_{\mathrm{ref},bg}$.

Источник: Eq. (4), с. 5; Algorithm 1, с. 14. Возможный batching current/reference forward или replicas не меняет логические суммы, но physical batch layout и numerical precision отсутствуют в статье.

### 3.7. Шаг S.7 — получить один learned $\log Z$ на prompt

**Inputs:** $x_b$, prompt hidden states и head $\phi_s$.

**Mathematical Operation:** $a_b=\operatorname{MLP}_{\phi_s}(\operatorname{Detach}(\operatorname{MeanPool}_{\mathrm{prompt}}H(x_b)))$.

**State Transformation:** сохранить один скаляр на prompt; этот скаляр используется во всех пяти residuals. Нельзя подать completion/reward или текущий rollout softmax вместо prompt features в авторскую head.

**Outputs:** $a_b=\log Z_{\phi_s}(x_b)$.

Источник: §4.1, с. 5; Algorithm 1, с. 14. Алгоритм не задаёт placement отдельного head forward по времени относительно S.5–S.6: S.7 фиксирует зависимости до residual, не дополнительный авторский порядок neural forward calls.

### 3.8. Шаг S.8 — собрать practical residual

**Inputs:** $S_{\theta,bg},S_{\mathrm{ref},bg},\widetilde r_{bg},\beta,a_b$.

**Mathematical Operation:**

$$\delta_{bg}=S_{\theta,bg}-S_{\mathrm{ref},bg}-\beta\widetilde r_{bg}+a_b.$$

**State Transformation:** сформировать $B\times5$ residuals; никаких explicit backward probabilities, group softmax weights, PPO ratio clipping, separate KL penalty или entropy bonus в этом loss нет.

**Outputs:** $\delta_{bg}$.

Источник: Algorithm 1, с. 14. Frozen reference встроен в score difference, не заменяет необходимость learned head.

### 3.9. Шаг S.9 — average squared residual по prompts и rollout

**Inputs:** полный residual batch.

**Mathematical Operation:**

$$\mathcal L_{\mathrm{TraFL}}=\frac1{5B}\sum_{b=1}^{B}\sum_{g=1}^{5}\delta_{bg}^{\,2}.$$

**State Transformation:** scalar objective для одной update; веса prompt и rollout в Algorithm 1 одинаковы. Score уже усреднён по masks внутри residual; loss здесь — **square of estimated residual**, а не отдельно averaged squares каждого mask residual.

**Outputs:** scalar loss.

Источник: Algorithm 1, с. 14. Последнее различие следует из порядка «estimate scores → compute δ → minimize δ²»; перенос square внутрь replicate sum меняет objective.

### 3.10. Шаг S.10 — ровно одна совместная update policy/head

**Inputs:** scalar loss, current trainable LoRA parameters $\theta_s$, head $\phi_s$, optimizer/scheduler state.

**Mathematical Operation:** минимизировать предыдущий batch loss по $(\theta,\phi)$ одним gradient update. Для фиксированных sampled completion, rewards, corruption и detached features раскрытие gradient указанного batch loss:

$$\nabla_\theta\mathcal L=\frac{2}{5B}\sum_{b,g}\delta_{bg}\nabla_\theta S_{\theta,bg},\qquad\nabla_\phi\mathcal L=\frac{2}{5B}\sum_{b,g}\delta_{bg}\nabla_\phi a_b.$$

Это **математическое раскрытие** Algorithm 1 с feature detachment из §4.1, не заданный авторами SGD optimizer. Reference/rewards/group means/mask draws не trainable; через $a_b$ gradient к prompt backbone features остановлен. Дискретный sampling rollout не заменён REINFORCE term: Algorithm 1 минимизирует empirical residual loss, а не описывает exact differentiation sampling distribution в Eq. (3). [EXPLICIT GAP IN PAPER: статья не даёт autograd-level pseudocode, подробности detach sampled tensors, differentiation-through-sampling и optimizer step formula; авторский дефолт отсутствует.]

**State Transformation:** обновить policy adapters и head; reference оставить frozen; новое состояние $(\theta_{s+1},\phi_{s+1})$. Exactly one update на текущий batch; next step получает fresh rollouts current policy. Постоянного replay buffer, old-policy ratio и нескольких optimization epochs в Algorithm 1 нет (App. A, с. 14).

**Outputs:** новая policy/head и завершённый training step.

Learning rate obeys specified peak/min cosine schedule; точная зависимость LR от $s$ и warmup не опубликована. Не вводить собственную closed-form schedule как авторскую (§4.1).

### 3.11. Шаг S.11 — условие остановки

**Inputs:** текущий step count $s$ и input $M$.

**Mathematical Operation:** повторять S.1–S.10 для $s=1,\ldots,M$. После update $M$ вернуть $p_\theta,Z_\phi$.

**State Transformation:** завершить run; выбрать checkpoint для evaluation по отсутствующему авторскому правилу.

**Outputs:** task-specific trained checkpoint $p_{\theta_{M+1}}$ и head $Z_{\phi_{M+1}}$, в нотации состояния до/после update из 2.5.

Источник: Algorithm 1, с. 14. [EXPLICIT GAP IN PAPER: конкретное M, early stopping, validation/checkpoint selection и save frequency не указаны; авторский дефолт отсутствует.]

### 3.12. Шаг E.1 — controlled evaluation и Pass@$k$

**Inputs:** task-specific trained checkpoint, base и доступные public baselines; выбранный benchmark; max length $\in\{256,512\}$; temperature. Применять одинаковый prompt format, decoding и correctness pipeline ко всем directly evaluated models (§4.2, с. 6; App. G, с. 23).

**Operation:** fastdLLM parallel sampler, block size 32, low-confidence strategy: на каждом denoising step выбирать positions для unmasking по confidence; stochastic nucleus sampling, top-$p=0.91$. Для main head-to-head использовать temperature 0.6; sweep TraFL/base $\{0.3,0.6,0.9\}$ (§4.2; §5.2).

**Outputs:** $n=16$ independently generated answers на problem, correctness count $c\in\{0,\ldots,16\}$; для math exact match extracted boxed answer, для code provided tests execution (§4.2; Figs. 1–2).

Авторская форма estimator:

$$\widehat{\operatorname{Pass@}k}=1-\prod_{i=n-c+1}^{n}\left(1-\frac{k}{i}\right).$$

Источник: §4.2, с. 6. **Математически эквивалентная комбинаторная форма**: $1-\binom{n-c}{k}/\binom nk$ при $1\leq k\leq n$, с $\binom{n-c}{k}=0$ при $n-c<k$. При $c=0$ empty product равен 1 и Pass@$k=0$; при $n-c<k$ estimator 1; Pass@1 равен $c/n$. Это условия математической формулы, не опубликованный numerical implementation. Усреднять problem-level estimates внутри benchmark; Fig. 1 average — по четырём benchmarks и двум lengths, не pooling 16 samples разных problems (Fig. 1, с. 7; Fig. 2, с. 8).

| Benchmark | Eval problems | Source / предназначение |
|---|---:|---|
| GSM8K | 1,319 | mathematical reasoning |
| MATH-500 | 500 | mathematical reasoning |
| Minerva Math, algebra split | 1,187 | held-out math, отдельно оба math-trained checkpoints |
| HumanEval | 164 | coding |
| MBPP | 500 | coding |
| LiveCodeBench release_v5 | 880 | held-out coding; easy/medium/hard и overall |

Источник: §4.2, с. 6. Path evaluation и judging не заменяют accuracy evaluation.

[EXPLICIT GAP IN PAPER: fastdLLM revision и полный sampler config, confidence thresholds/tie-breakers, KV-cache settings, number of maximum denoising steps для основного evaluation, seeds, evaluator versions, error bar / standard error calculation и split counts LiveCodeBench не указаны; авторский дефолт отсутствует.]

### 3.13. Шаг E.2 — path-diversity evaluation

**Inputs:** prompt, reference и сравниваемые policies; $N=16$ independent samples, $L=256$, $T_{\mathrm{steps}}=256$.

**Operation:** на каждом $t=1,\ldots,256$ записать $S_t^i$ — set token positions, unmasked в sample $i$. Для каждой unordered pair $1\leq i<j\leq16$ и step вычислить $J_t^{ij}=|S_t^i\cap S_t^j|/|S_t^i\cup S_t^j|$; average across pairs и steps. Явная запись того же average, когда denominators определены:

$$\bar J(x)=\frac1{256\binom{16}{2}}\sum_{t=1}^{256}\sum_{i<j}J_t^{ij}(x),\qquad\mathrm{Preserved}=100\%\times\frac{\bar J_{\mathrm{ref}}}{\bar J_{\mathrm{model}}}.$$

**Outputs:** Jaccard-based preserved percentage; высокий $\bar J$ означает повторение same-step schedule и locking, низкое preserved — loss of diversity. Это ratio inverse overlap, не Shannon entropy или число distinct terminal modes (§5.1, с. 7).

[EXPLICIT GAP IN PAPER: convention для S_t^i∪S_t^j=∅, early termination/padding шага, J̄model=0, точные prompts/count и aggregation across prompts, evaluation temperature и sampler overrides конкретно path analysis не указаны; авторский дефолт отсутствует.]

Контроль Tab. 1, с. 7 (HE = HumanEval; «MATH», «GSM» — labels таблицы):

| Model | HE | MBPP | MATH | GSM | Avg. preserved (%) |
|---|---:|---:|---:|---:|---:|
| TraFL | 98 | 95 | 104 | 109 | 101 |
| ESPO | 89 | 86 | 95 | 79 | 87 |
| JustGRPO | 59 | 58 | 62 | 47 | 56 |

Значения выше 100% допустимы для ratio: авторы трактуют average 101% как reference-level preservation, не доказанную diversity gain (§5.1).

### 3.14. Шаг E.3 — normalization ablations

**Inputs:** MATH train run; все прочие components, hyperparameters и training data фиксированы.

**Operation:** сравнить три варианта: (i) **Constant Z**: $\log Z=0$ fixed; (ii) **Batch-estimated Z**: оценивать $\log Z$ на current rollout group each step; (iii) **Learned Z**: описанная prompt-dependent MLP. Evaluation: Pass@5, $n=16$, temperature 0.6, MATH-500 и Minerva algebra, lengths 256/512 (App. B, с. 14–15).

**Outputs:** Tab. 2 values:

| Normalization | MATH-500 256 | MATH-500 512 | Minerva 256 | Minerva 512 | Avg. |
|---|---:|---:|---:|---:|---:|
| Constant | 51.1 | 56.5 | 69.5 | 75.8 | 63.2 |
| Batch-estimated | 52.7 | 55.9 | 69.5 | 72.5 | 62.7 |
| Learned TraFL | 54.6 | 57.1 | 70.7 | 75.8 | 64.6 |

[EXPLICIT GAP IN PAPER: точная batch-estimated logZ формула, detach/gradient convention, use raw или centered reward и implementation соответствия DMPO для ablation не указаны; авторский дефолт отсутствует.]

**Отдельно анализ DMPO, не готовая ablation-рецептура:** App. D.6, с. 20–21 определяет $w^{(n)}=e^{\ell^{(n)}}/\sum_k e^{\ell^{(k)}}$, $\ell^{(n)}=r^{(n)}/\alpha_{\mathrm{DMPO}}+\log[p_{\mathrm{ref}}(o^{(n)}\mid x)/p_v(o^{(n)}\mid x)]$, weighted DCE objective $\sum_nw^{(n)}L_{\mathrm{DCE}}(o^{(n)})$ со stop-gradient weights. Буфер logits $\varphi^{(n)}=\log p_\theta(o^{(n)}\mid x)$ предполагают additive flow $\dot\varphi^{(n)}\propto w^{(n)}$; conditional $q^{(n)}=p_\theta(o^{(n)}\mid x)/\sum_kp_\theta(o^{(k)}\mid x)$ obeys $\dot q^{(n)}=q^{(n)}(w^{(n)}-\bar w)$ и $q^{(n)}(t)=q^{(n)}(0)e^{w^{(n)}t}/\sum_kq^{(k)}(0)e^{w^{(k)}t}$ в данном аналитическом режиме. Высокий weight усиливается, absent-buffer completion не получает direct gradient. Это обсуждение batch-softmax limitation, не значение $\beta$ TraFL и не вычисление $\log Z_\phi$.

### 3.15. Шаг E.4 — LLM-as-judge diversity

**Inputs:** MATH-500 и LiveCodeBench problems; outputs generated max length 256, temperature 0.6, sets of 16 для двух методов. Judge GPT-5-4; AI use statement называет его GPT-5.4. Judge blind to method identity; randomize A/B answer-set order (App. H, с. 23; §5.5, с. 9; AI use statement, с. 10).

**Operation:** для pairs TraFL/Base, ESPO/Base, TraFL/ESPO получить categorical decision $S_{\mathrm{LLM}}(D_A,D_B)\in\{A_{\mathrm{wins}},B_{\mathrm{wins}},\mathrm{Tie}\}$. Не включать JustGRPO в prespecified hypotheses: его judge comparison в Tab. 5 нет. Оценивать distinct underlying approaches, не superficial diversity (App. H–I).

Два режима: **all samples**; **correct solutions**, где сначала filter each side to correct answers и retain problem comparison, только если обе стороны имеют хотя бы одно correct solution. Correct-solution filtering делает размеры subset переменными, хотя исходные sets содержат 16. В основном тексте primary analysis именно correct-solution (§5.5; App. H, с. 24).

**Текст judge prompt из App. I, с. 25**, без программного кода; placeholders заменяются problem и двумя answer sets:

> [Judge Role]
> You are an expert judge evaluating diversity among LLM-generated answers to the same problem.
> Your goal is to assess approach diversity, not correctness or answer quality.
>
> [Problem]
> {problem}
>
> [Set A]
> {set_a}
>
> [Set B]
> {set_b}
>
> [Definition of Distinct Approach]
> An answer represents a distinct approach only if it uses a recognizably different core method, such as a different algorithm, data structure, mathematical reduction, equation setup, case analysis, proof strategy, or reasoning path.
>
> [Do Not Count as Diversity]
> – different wording, formatting, verbosity, or variable names
> – different final answers from the same method
> – minor bugs in the same method
> – random guesses or incoherent reasoning
> – extra explanation that does not change the core method
>
> [Instructions]
> 1. Identify the distinct approach types in each set.
> 2. Compare the number and substance of these approach types.
> 3. Prefer TIE if the difference is mostly superficial or unclear.

Заключительный **[Response Format]** требует “Return exactly this JSON:” и поля **set_a_approaches**, **set_b_approaches** (arrays коротких имён distinct approaches), **winner** $\in\{\text{A},\text{B},\text{TIE}\}$, **confidence** $\in\{\text{low},\text{medium},\text{high}\}$, **reason** (one concise sentence). Здесь response schema передана словами, без сериализованного кода; точная JSON-пунктуация находится в source App. I, с. 25.

**Statistics:** заранее заданные directional alternatives TraFLwins > Basewins, ESPOwins > Basewins, TraFLwins > ESPOwins отдельно для каждого benchmark. One-sided **exact sign tests over non-ties**, threshold $p\leq0.05$; For/Against/Tie — доли judged problems, не голосов среди одних non-ties (App. H, с. 23–24). **Математическая расшифровка теста:** если $f$ wins for alternative и $a$ против при $u=f+a$, one-sided $p=\sum_{j=f}^{u}\binom uj2^{-u}$. Статья не задаёт поведения $u=0$ и multiple-comparison adjustment.

| Benchmark / alternative | All: For / Against / Tie (%) | All: $p$ | Correct: For / Against / Tie (%) | Correct: $p$ |
|---|---|---|---|---|
| MATH-500 TraFL > Base | 21.8 / 15.4 / 62.8 | $2.3\times10^{-2}$ | 12.4 / 5.1 / 82.5 | $6.0\times10^{-3}$ |
| MATH-500 ESPO > Base | 18.2 / 17.6 / 64.2 | $8.8\times10^{-1}$ | 7.5 / 10.0 / 82.6 | $3.9\times10^{-1}$ |
| MATH-500 TraFL > ESPO | 21.8 / 14.2 / 64.0 | $5.7\times10^{-3}$ | 14.3 / 5.6 / 80.1 | $1.0\times10^{-3}$ |
| LiveCodeBench TraFL > Base | 20.9 / 23.0 / 56.1 | $3.9\times10^{-1}$ | 31.2 / 19.3 / 49.6 | $4.1\times10^{-5}$ |
| LiveCodeBench ESPO > Base | 14.5 / 44.2 / 41.3 | $1.0\times10^0$ | 31.1 / 34.0 / 34.9 | $7.9\times10^{-1}$ |
| LiveCodeBench TraFL > ESPO | 45.9 / 12.2 / 41.9 | $<10^{-4}$ | 38.3 / 26.4 / 35.3 | $7.8\times10^{-4}$ |

Источник: Tab. 5, с. 24. Сохранены напечатанные rounding и $p$; не пересчитывать statistical significance из округлённых percentages. Significant all-samples: TraFL>Base MATH, TraFL>ESPO оба; significant correct: TraFL>Base и TraFL>ESPO оба; ESPO>Base не significant.

**Внутренняя несогласованность источника, а не исправление чисел:** для заявленного one-sided alternative «For > Against» upper-tail formula выше при $f<a$ даёт $p>0.5$. Однако Tab. 5 печатает $p=0.39$ для MATH-500 / correct / ESPO>Base при For 7.5% < Against 10.0%, и $p=0.39$ для LiveCodeBench / all / TraFL>Base при For 20.9% < Against 23.0%. Эти направления не объясняются округлением. [EXPLICIT GAP IN PAPER: заявленный directional one-sided sign-test protocol и некоторые опубликованные p-values внутренне несогласованы; actual calculation procedure и raw counts не приложены; авторское разрешение противоречия отсутствует.] Формула описывает **заявленный** тест, но не обещает воспроизвести все напечатанные $p$; значения source table оставлены без изменения (App. H, с. 23–24).

[EXPLICIT GAP IN PAPER: exact model snapshot/API settings judge, temperature/token limit judge itself, randomization seeds, answer-set serialization/response retries, problem counts после correct filtering, raw wins/ties и все malformed-judge-output rules не указаны; авторский дефолт отсутствует.]

### 3.16. Контрольные accuracy results и baselines

Эти цифры — авторские outcomes, не hyperparameters и не acceptance threshold для слабейшей модели. Direct baselines: public checkpoints JustGRPO и ESPO, общая evaluation pipeline. Math у каждого: separate GSM8K (7,473) / MATH (7,500) training. Code у public JustGRPO и ESPO: AceCode-89K, **21,000 hard samples**; у TraFL: filtered mixture 14,632. Это data mismatch, а не утверждение о same-data controlled coding training (Tab. 3, с. 23; App. F.2).

Pass@5 %, $n=16$, temperature 0.6, Fig. 2, с. 8:

| Method / max length | GSM8K | MATH-500 | HumanEval | MBPP |
|---|---:|---:|---:|---:|
| Base / 256 | 91.3 | 50.2 | 53.2 | 59.1 |
| JustGRPO / 256 | 89.7 | 50.4 | 47.1 | 63.1 |
| ESPO / 256 | 91.7 | 54.6 | 49.5 | 61.3 |
| TraFL / 256 | 92.6 | 54.6 | 54.8 | 62.1 |
| Base / 512 | 92.0 | 56.3 | 53.7 | 59.2 |
| JustGRPO / 512 | 89.6 | 53.9 | 46.4 | 65.0 |
| ESPO / 512 | 91.6 | 59.0 | 55.0 | 61.5 |
| TraFL / 512 | 92.3 | 57.1 | 55.7 | 62.0 |

Fig. 2 reports aggregate average Pass@5 Base 64.4, JustGRPO 63.2, ESPO 65.5, TraFL 66.4; average denoising steps respectively **121, 66, 100, 118**. Error bars standard error. Это outcomes, не фиксированные generation step limits; average denoising steps не заменяет $T_{\mathrm{steps}}=256$ отдельного path protocol (§5.3, с. 8).

Minerva algebra Pass@5, Fig. 3(a), с. 9:

| Method | GSM8K-trained / 256 | MATH-trained / 256 | GSM8K-trained / 512 | MATH-trained / 512 |
|---|---:|---:|---:|---:|
| Base | 68.7 | 68.7 | 75.1 | 75.1 |
| JustGRPO | 64.2 | 64.6 | 72.6 | 66.7 |
| ESPO | 71.2 | 71.4 | 75.3 | 75.6 |
| TraFL | 69.5 | 70.7 | 76.3 | 75.8 |

LiveCodeBench Pass@5, Fig. 3(b), с. 9:

| Method / max length | Easy | Medium | Hard | Overall |
|---|---:|---:|---:|---:|
| Base / 256 | 49.7 | 61.3 | 48.8 | 53.8 |
| JustGRPO / 256 | 42.1 | 56.3 | 34.8 | 45.2 |
| ESPO / 256 | 48.9 | 59.1 | 41.5 | 50.5 |
| TraFL / 256 | 55.4 | 69.2 | 62.4 | 62.6 |
| Base / 512 | 59.4 | 68.0 | 55.5 | 61.4 |
| JustGRPO / 512 | 44.0 | 57.8 | 38.9 | 47.6 |
| ESPO / 512 | 53.0 | 68.0 | 50.4 | 57.8 |
| TraFL / 512 | 65.6 | 71.5 | 60.7 | 66.3 |

D1 и DMPO: checkpoints не были доступны авторам, их опубликованные single-sample accuracy лишь **quoted contextual reference**, не independently reproduced direct baselines. Сами алгоритмы D1/DMPO/JustGRPO/ESPO не полностью специфицированы этой статьёй (§4.2; App. G, с. 23).

| Context-only method | GSM8K 256 / 512 | MATH-500 256 / 512 | HumanEval 256 / 512 | MBPP 256 / 512 | Avg. |
|---|---|---|---|---|---|
| D1† | 81.1 / 82.1 | 38.6 / 40.2 | 32.9 / 37.8 | 44.7 / 42.8 | 50.03 |
| DMPO† | 82.41 / 85.22 | 38.20 / 42.80 | — / — | — / — | — |

Источник: Tab. 4, с. 23. Не сравнивать эти quoted single-sample values как собственные controlled Pass@5.

### 3.17. Вычислительные и data-release ограничения источника

App. J, с. 25: 8 modern parallel accelerator cards, минимум 80 GB memory; примерно 10 часов **на один math experiment**, 18 часов code. Exact accelerator model, precision, implementation efficiency и число updates из часов не следуют.

App. K, с. 25 перечисляет licensing используемых authors releases: MIT для GSM8K, MATH, HumanEval, AceCode-89K, указанного public Minerva release и LLaDA-8B-Instruct; CC BY 4.0 для MBPP; CC BY-NC 4.0 и non-commercial research для KodCode-Light-RL-10K; Creative Commons license family для LiveCodeBench v5, используемого только evaluation. Это атрибуция статьи, не проверка лицензий новых пользовательских releases/checkpoint. Reproducibility statement обещает source code/trained checkpoints **upon acceptance**, а не прикладывает их к данному PDF (с. 10).

## 4. Edge Cases & Numerical Stability Rules

### 4.1. Правила, действительно заданные авторами

| Ситуация | Авторское правило | Что это гарантирует / не гарантирует |
|---|---|---|
| Variance current–reference score | identical masking pattern | уменьшает variance разности, не approximation bias (Eq. (4); App. C, с. 16) |
| Mask Monte Carlo variance | four antithetic replicates, complementary pairs | точный coupling не опубликован (§4.1, с. 5) |
| EOS tokens | mask только non-EOS | domain restriction, не полноценная спецификация length и empty cases (§4.1) |
| Reward offsets | center внутри prompt group | heuristic variance reduction, не exact deterministic-Z absorption (App. D.5) |
| Head gradient | detached backbone prompt features | исключает gradient head к backbone по feature input (§4.1) |
| Reference drift | frozen reference | reference weights не меняются (§3.2) |
| Update reuse | fresh online rollouts, one update | следующий batch generated новой current policy (App. A) |
| Mask count формальной Eq. (4) | $l\geq1$ | исключает $1/0$ **в формальном score**, не разрешает antithetic empty complement (Eq. (4)) |

### 4.2. Математически определённые границы

При all-success или all-failure training group centering даёт zero reward term; остаются score difference и learned $a_b$. В Algorithm 1 нет skip rule и нет std denominator. Centered reward бывает отрицательным: это предусмотренное $r-\bar r$, не numerical error. Нельзя clamp centered reward к $[0,1]$ (Algorithm 1, с. 14).

Formal mask size $l=L$ означает fully masked completion; formal Eq. (4) определён при $L>0$. Но complement того же set имеет zero positions и не удовлетворяет $l\in\{1,\ldots,L\}$: статья не разрешает это противоречие complementary replicate construction (§4.1; Eq. (4)).

Формальные log probabilities требуют probability $>0$ для finite log. Если probability 0, $\log0=-\infty$, current/ref difference может быть неопределённым; никакой epsilon, clamp, log-softmax precision или finite-loss recovery rule авторы не задают (Eq. (4); Algorithm 1). Это граница математического выражения, не инструкция вводить собственные constants.

Pass@$k$ имеет domain $1\leq k\leq n$; empty product и zero binomial conventions указаны в 3.12 как mathematical consequences. Не распространять reported estimator на $k>n$, $n=0$ или отрицательные counts (§4.2, с. 6).

### 4.3. Неопределённые случаи, которые нельзя заполнить авторским дефолтом

- [EXPLICIT GAP IN PAPER: пустой completion L=0 и completion без non-EOS positions не имеют опубликованного scoring/reward/skip правила; авторский дефолт отсутствует.]
- [EXPLICIT GAP IN PAPER: пустой prompt batch B=0 не имеет правила, при этом batch mean loss делит на B; авторский дефолт отсутствует.]
- [EXPLICIT GAP IN PAPER: complementary mask нулевого размера, нечётная eligible length, rounding antithetic levels и EOS внутри/после окончания sequence не имеют правила; авторский дефолт отсутствует.]
- [EXPLICIT GAP IN PAPER: padding, длины при truncated generation, special-token exclusion кроме EOS, residual mask token в final y и prompt special tokens не специфицированы; авторский дефолт отсутствует.]
- [EXPLICIT GAP IN PAPER: log(0), NaN/Inf, mixed precision, loss/gradient scaling, gradient clipping, optimizer overflow/skipped updates и Z saturation не имеют авторских numerical rules; авторский дефолт отсутствует.]
- [EXPLICIT GAP IN PAPER: negative/very large learned logZ не имеют clamp/regularization/range rule; авторский дефолт отсутствует.]
- [EXPLICIT GAP IN PAPER: empty Jaccard union, zero J̄model и unequal effective denoising duration не имеют авторского averaging convention; авторский дефолт отсутствует.]
- Пустые correct-solution sets **исключают comparison**, если хотя бы одна сторона не содержит correct solution; это заданное правило, не GAP (App. H, с. 24).
- [EXPLICIT GAP IN PAPER: crash/timeout rewards, malformed extracted math/code answers, malformed judge response, нулевое число retained correct-solution comparisons и sign test без non-ties не имеют полного recovery protocol; авторский дефолт отсутствует.]

## 5. Identified Ambiguities & Paper Gaps

### 5.1. Пробелы, блокирующие единственную авторскую реализацию

| ID | Неопубликованная деталь | Что статья реально задаёт | Последствие для точного воспроизведения |
|---|---|---|---|
| G01 | [EXPLICIT GAP IN PAPER: численное β отсутствует; авторский дефолт отсутствует.] | $\beta>0$ Eq. (1); input Algorithm 1 | неизвестен reward/score scale |
| G02 | [EXPLICIT GAP IN PAPER: M, число epochs/updates отсутствуют; авторский дефолт отсутствует.] | loop 1…M | training duration в updates не восстановить из wall time |
| G03 | [EXPLICIT GAP IN PAPER: prompt batch size и data sampling strategy отсутствуют; авторский дефолт отсутствует.] | sample $D_b\sim D$ | не задана effective batch / coverage |
| G04 | [EXPLICIT GAP IN PAPER: optimizer, momentum/betas, epsilon, weight decay и optimizer head разделение отсутствуют; авторский дефолт отсутствует.] | jointly minimize loss, peak/min LR | одинаковый loss не определяет weights trajectory |
| G05 | [EXPLICIT GAP IN PAPER: warmup, exact cosine horizon/phase и scheduler step placement отсутствуют; авторский дефолт отсутствует.] | cosine decay | LR per update не определён |
| G06 | [EXPLICIT GAP IN PAPER: LoRA targets, init, dropout/bias, trained extra modules отсутствуют; авторский дефолт отсутствует.] | rank128, alpha128 | неизвестны trained parameter set / initial policy change |
| G07 | [EXPLICIT GAP IN PAPER: reference/init exact revisions и construction relation отсутствуют; авторский дефолт отсутствует.] | initial policy + frozen ref, base LLaDA-8B-Instruct | reference equality to loaded init не доказана как low-level recipe |
| G08 | [EXPLICIT GAP IN PAPER: 2-layer MLP hidden dimensions/activation/bias/init отсутствуют; авторский дефолт отсутствует.] | prompt-dependent scalar logZ head | normalization dynamics неоднозначны |
| G09 | [EXPLICIT GAP IN PAPER: prompt hidden layer/source backbone/pooling tokens и feature recomputation timing отсутствуют; авторский дефолт отсутствует.] | mean pooling + detach | head input не однозначен |
| G10 | [EXPLICIT GAP IN PAPER: training sampler параметры и stopping rules отсутствуют; авторский дефолт отсутствует.] | 5 current-policy rollout | нельзя использовать evaluation settings как авторские training |
| G11 | [EXPLICIT GAP IN PAPER: exact antithetic coupling/levels/weights отсутствуют; авторский дефолт отсутствует.] | K=4 total replicates, complementary masks | endpoint/variance/expectation scheme не определены |
| G12 | [EXPLICIT GAP IN PAPER: relation formal L to non-EOS eligible set и scoring empty cases отсутствует; авторский дефолт отсутствует.] | Eq.(4) vs §4.1 | нельзя однозначно реализовать corruption domain |
| G13 | [EXPLICIT GAP IN PAPER: autograd-level handling sampling и rewards отсутствует; авторский дефолт отсутствует.] | empirical loss minimization; detach head features | conditional gradients в 3.10 — раскрытие формулы, не полный код авторов |
| G14 | [EXPLICIT GAP IN PAPER: gradient accumulation/distribution/reduction/precision/clipping отсутствуют; авторский дефолт отсутствует.] | one update, compute notes | hardware recipe не определена |
| G15 | [EXPLICIT GAP IN PAPER: reward extraction/tests/timeouts/recovery отсутствуют; авторский дефолт отсутствует.] | binary exact match / tests | final training rewards могут отличаться |
| G16 | [EXPLICIT GAP IN PAPER: data revisions/retained IDs/filtering thresholds и weaker KodCode proxy отсутствуют; авторский дефолт отсутствует.] | filtering outline and exact sizes | same subset не восстановить по размерам |
| G17 | [EXPLICIT GAP IN PAPER: exact prompt serialization/chat template отсутствует; авторский дефолт отсутствует.] | representative prompts, no system | rendered text не равен tokenizer input specification |
| G18 | [EXPLICIT GAP IN PAPER: batch-estimated Z ablation formula отсутствует; авторский дефолт отсутствует.] | rollout-local estimate like DMPO | numerical ablation cannot be reproduced uniquely |
| G19 | [EXPLICIT GAP IN PAPER: checkpoint selection/save/resume/validation rules и seeds отсутствуют; авторский дефолт отсутствует.] | return trained policy | evaluated weights не однозначны |
| G20 | [EXPLICIT GAP IN PAPER: full evaluation fastdLLM settings/version/steps/thresholds/tie-breakers отсутствуют; авторский дефолт отсутствует.] | sampler name, block32, top-p0.91, T/lengths | shared high-level decoding не задаёт exact implementation |
| G21 | [EXPLICIT GAP IN PAPER: evaluation pipeline versions/error bars/LCB split counts отсутствуют; авторский дефолт отсутствует.] | dataset counts, tests, standard error label | exact metrics/uncertainty не восстановить |
| G22 | [EXPLICIT GAP IN PAPER: path analysis prompt aggregation/empty union/termination/temperature отсутствуют; авторский дефолт отсутствует.] | 16 samples,256 tokens,256 steps,Jaccard | preserved percentages неоднозначны на edge cases |
| G23 | [EXPLICIT GAP IN PAPER: exact judge model snapshot/settings/serialization/retries/randomization и raw eligible counts отсутствуют; авторский дефолт отсутствует.] | prompt, blind/random order, tests and percentages | exact judge outcomes и p-values не восстановить |
| G24 | [EXPLICIT GAP IN PAPER: полные training алгоритмы baselines и checkpoint identifiers отсутствуют; авторский дефолт отсутствует.] | public JustGRPO/ESPO, contextual D1/DMPO | чужие methods требуют собственных источников |

### 5.2. Смысловые ограничения: нельзя обещать то, чего paper не доказывает

**Surrogate versus exact likelihood.** Eq. (4) — score lower-bound relation uniform reveal-order model, не true sampler likelihood. Нельзя представлять его как unbiased policy log probability, менять $\beta$ для «компенсации length» без авторского указания или объявлять exact trajectory balance achieved (App. C, с. 15–16; App. D.5).

**Centering versus target.** Algorithm 1 использует centered group rewards, Eqs. (1)–(3) raw terminal rewards. Random group mean нельзя точно спрятать в deterministic learned head. Поэтому numerical predictor $Z_\phi$ нельзя автоматически идентифицировать с true partition function raw target (App. D.5, с. 20).

**Gradient graph versus exact expectation gradient.** Fixed-batch gradient squared residual и полная производная theoretical expectation Eq. (3) — разные математические операции. Paper Algorithm 1 задаёт первую последовательность вычислений; отдельный policy-gradient-through-sampling член не описан. Spec не подменяет empirical training этим дополнительным членом и отмечает отсутствие low-level graph details (App. A).

**Path invariance versus immutable conditionals.** Equal expected score для same completion не означает $p_\theta(\tau\mid y,x)=p_{\mathrm{ref}}(\tau\mid y,x)$ на каждом neural update. Shared parameters могут менять conditionals indirectly; preservation проверяется §5.1 (App. D.5).

**Pure reward versus replicator auxiliary theory.** Eq. (11) зависит от additive payoff assumption Eq. (9), не следует из обычного Euclidean policy gradient Eq. (8). Proposition 2 ограничена tabular softmax и positive reward advantage; единственная trajectory не доказана как inevitable neural limit (App. D.2–D.3).

**Conceptual backward TB versus actual reference tilt.** Eq. (14) и reference-weighted completion marginal Eq. (1) различны. Их нельзя объединять в одну exact implementation identity (App. D.4–D.5).

**Diversity measures.** Jaccard schedules, support/entropy theorem, all-samples judge и correct-solution judge — разные измерения. Higher Pass@$k$ alone не доказывает broader mode coverage; статья поэтому отдельно использует judge (§5.1; §5.5; App. H).

**Protocol versus expected outcomes.** Авторская модель instruction-tuned, SFT сверх неё отсутствует. Единственное заявленное пользовательское изменение — **Qwen 0.8B dLLM**, diffusion-модель с более слабыми исходными весами. Точный checkpoint и уровень instruction tuning не сообщены. Авторский masked-denoising conditional $p_\theta(y_i\mid y_l,x)$ и diffusion sampling сохраняются; дополнительная SFT и замена score другими likelihood не входят в задание. LoRA rank128/alpha128 сохраняются как опубликованные значения; их target modules и вход head надо привязать к фактической архитектуре без выдуманных внутренних размеров (§4.1; пользовательское уточнение).

### 5.3. Что требуется для доведения до буквального воспроизведения

Получить от авторов code/checkpoint/config и data-filter outputs для G01–G24 либо зафиксировать каждый недостающий выбор как **собственное явно маркированное отклонение**, отдельно от этой спецификации. В PDF утверждение «all hyperparameters are reported» (Reproducibility statement, с. 10) не заменяет конкретные отсутствующие значения. До их появления корректный результат — faithful reconstruction всех опубликованных шагов и честный перечень неопределённостей, а не обещание полностью определённой авторской реализации.
