# Что именно меняется: RTB, completion score, ELBO, маски и центрирование

2026-10-10. Эта записка отвечает на конкретный вопрос: какие части практического
TraFL являются заменой цели, какие — приближением score, какие — stochastic
estimation, а какие — отдельными решениями обучения. Сопутствующий
[FORMULATIONS_V2.md](FORMULATIONS_V2.md) даёт определения и implementation map.
Первая версия FORMULATIONS.md снята (в ней не было поправки на центрирование).

**Основной эмпирический факт не оставлен «открытым».** В сохранённых нейронных
запусках conditional path distributions действительно изменялись (§7).
Открыты обобщение и причинное объяснение, а не существование этих наблюдений.
Равенство losses, unbiased score, unbiased loss estimate, expected gradient
и успешный trained endpoint — пять разных утверждений.

## 1. Карта переходов: не одно «приближение TraFL»

Для начала держим фиксированными reference, reward, prompt, длину,
единицы score, декодер и способ sampling. Центрирование пока не используем.

| Шаг | Объект, который скорится | Что меняется относительно предыдущего шага |
|---|---|---|
| A. Exact RTB-style relative path loss | $\log P_\theta(\tau\mid x)-\log P_0(\tau\mid x)$ | Исходная совместная reference-tilted цель; нужны правильные probabilities полного пути |
| B. Exact completion loss | $\log p_\theta(y\mid x)-\log p_0(y\mid x)$ | Убирается прямое различение путей к тому же y. Это смена objective, а не несмещённая оценка A |
| C. Exact mask-mean loss | Разность двух ELBO, в intensive units | Заменяется marginal score. Даже без Monte Carlo появляется другой conditional-order term |
| D. Finite IID mask square | Квадрат среднего K масочных score ratios | Добавляется ожидаемая mask-variance penalty и stochastic gradient noise |
| E. Biased/dependent masks | Иное распределение масок, например local comp | Может измениться и mean score, и variance. Зависимость требует covariance accounting |
| F. IID product estimator | Произведение независимых residual means | Убирается ожидаемая mask-variance penalty, но не C и не stochastic gradient noise |

**Отдельные оси:** group reward centering, advantage scaling, learned/profiled
normalizer, ratio centering, rollout decoder, actor family, optimizer, clipping,
schedule, sample count и forward cost. Они не становятся эквивалентными от
одинакового названия метода или параметра beta.

## 2. Exact RTB: что зафиксировано и что является целью

Дальше $u>0$ — число скрытых позиций одного фиксированного prompt x,
$\sigma$ — reveal order, $y$ — полное завершение и
$\tau_\sigma(y)$ — весь путь к нему. Обозначим

$$
P_\theta(\sigma,y\mid x)=p_\theta(y\mid x)c_\theta(\sigma\mid x,y).
$$

Все P и p относятся к **одному объявленному декодеру**.
Reference P0 заморожен. Для scalar reward r и конечного Z:

$$
Q(\sigma,y\mid x)=P_0(\sigma,y\mid x)e^{\beta r(y)}/Z(x).
$$

Тогда target terminal $q=p_0e^{\beta r}/Z$, а target conditional равен c0.
Если r=logR, это $p_0R^\beta$, не $p_0\exp(\beta R)$.
Если r является utility, включая бинарную, используется именно эта utility;
универсальной команды «взять log reward» нет.

Exact intensive path residual:

$$
\delta_{\rm path}(\sigma,y)
=\frac{\log P_\theta(\sigma,y\mid x)-\log P_0(\sigma,y\mid x)-\beta r(y)}{u}
+z(x).
$$

При идеальном равенстве Pθ=Q значение z=logZ/u. Его кодовое имя log_z
не доказывает, что обученный head является настоящим partition function.

### 2.1 Почему один scalar для разных путей

Сначала фиксируем y, а не порядок. Например, одна строка 010 получается
по путям □□□→0□□→01□→010 и □□□→□□0→□10→010.
Промежуточные strings определяют prediction contexts, но score Xσ — сумма
по **всей** траектории:

$$
X_\sigma(y)=u^{-1}\log[P_\theta(\tau_\sigma(y)\mid x)/P_0(\tau_\sigma(y)\mid x)].
$$

Сдвиг $b(y)=\beta r(y)/u-z(x)$ одинаков для всех порядков к данному y:
reward tilt зависит от завершения. Требуется одинаковое относительное
reweighting путей, а не равенство их абсолютных probabilities и не следование
одному «reference order».

## 3. Замена пути точным завершением: разные objectives

Введём

$$
D_y=\frac{\log p_\theta(y\mid x)-\log p_0(y\mid x)}{u},\qquad
L_\sigma(y)=\log\frac{c_\theta(\sigma\mid x,y)}{c_0(\sigma\mid x,y)},\qquad
a(y)=D_y-\beta r(y)/u+z(x).
$$

Тогда **точно**

$$
\delta_{\rm path}=a(y)+L_\sigma(y)/u,\qquad
\delta_{\rm comp}=a(y).
$$

Completion scoring не является estimator path scoring, у которого просто
меньше шума. Он выбрасывает член, различающий conditional allocation.
При exact terminal fit возможны разные cθ; shared neural parameters могут
изменять cθ при обучении pθ.

### 3.1 Какой закон используется при averaging по порядкам

Одна и та же алгебра bias–variance decomposition имеет три разных применения:

$$
E_\nu[\delta_{\rm path}^2\mid y]
=\left(a(y)+E_\nu L_\sigma/u\right)^2
+\operatorname{Var}_\nu(L_\sigma)/u^2.
$$

| Order measure при фиксированном y | $E_\nu L_\sigma$ |
|---|---|
| Actual on-policy conditional $c_\theta$ | $\mathrm{KL}(c_\theta\Vert c_0)$ |
| Reference conditional $c_0$ | $-\mathrm{KL}(c_0\Vert c_\theta)$ |
| Uniform orders U | $\mathrm{KL}(U\Vert c_0)-\mathrm{KL}(U\Vert c_\theta)$ |

Следовательно on-policy expected path square содержит и mean shift, и
variance, а не просто completion square плюс одна variance penalty.

Uniform position selection **до conditioning на y** не означает uniform
conditional orders после conditioning. Схема «sample y от actor, затем
force uniform orders к нему» задаёт меру pθ(y)U(σ), а не actual actor measure
pθ(y)cθ(σ|y). Это может быть корректным отдельным диагностическим объектом,
но его нельзя молча назвать on-policy RTB.

### 3.2 Expected updates: точное различие при сильных предпосылках

Роллауты в коде являются detached data. Если path/completion probabilities
точно нормированы, reference имеет нужный support, sampling fresh on-policy,
z prompt-only и не зависит от actor parameters, то expected intensive
actor semigradients без centering:

$$
g_{\rm path}=\frac{2}{u^2}\nabla[
\mathrm{KL}(P_\theta\Vert P_0)-\beta E_\theta r],
\qquad
g_{\rm comp}=\frac{2}{u^2}\nabla[
\mathrm{KL}(p_\theta\Vert p_0)-\beta E_\theta r].
$$

Их различие:

$$
g_{\rm path}-g_{\rm comp}
=\frac{2}{u^2}\nabla E_{p_\theta}
\mathrm{KL}(c_\theta\Vert c_0).
$$

Это сильнее, чем интуиция о variance: различаются expected update fields.
Это не полный gradient expected on-policy squared loss, который ещё включал
бы производную sampling measure. При extensive path square множителя 1/u²
нет; одинаковый LR с intensive square не означает одинаковую оптимизацию.
Если u меняется между prompts, меняется и relative prompt weighting.

Этот вывод не доказывает neural convergence, достижимость Q или идентичность
Adam/clipped/multi-epoch updates. PG без reference имеет иной update и не
несёт этой conditional-KL regularization; это само по себе не theorem
неизбежного locking любой neural parameterization.

## 4. Замена exact completion score на ELBO: gap остаётся без sampling noise

Здесь декодер специально uniform-position serial, температура 1.
Пусть U — uniform measure на u! порядках и
$J_\theta(y)=\mathrm{KL}(U\Vert c_\theta(\cdot\mid x,y))$.

Для каждой сети

$$
\mathrm{ELBO}_\theta(y\mid x)=\log p_\theta^U(y\mid x)-J_\theta(y).
$$

Exact mask-mean ratio с count-uniform weights:

$$
D_E(y)=\frac{\mathrm{ELBO}_\theta-\mathrm{ELBO}_0}{u}
=D_y+\frac{J_0-J_\theta}{u}=E_U X_\sigma(y).
$$

Поэтому

$$
\delta_E=a(y)+(J_0-J_\theta)/u.
$$

Полное перечисление масок убирает Monte Carlo error при оценке DE.
Оно не убирает J-gap и не превращает objective в exact completion matching.
При данном y изменение pθ, conditional consistency и z могут компенсировать
друг друга в residual. Это возможность, не универсально достигаемый neural
fixed point: эти quantities связаны parameterization и decoder constraints.

Две lower bounds не дают lower bound их разности. DE может даже иметь другой
знак, чем Dy. Для confidence/parallel sampler masked DE вообще не объявляется
ELBO его собственного marginal pD без отдельного вывода.

Полезное точное тождество:

$$
E_U[\delta_{\rm path}^2\mid y]
=\delta_E(y)^2+\operatorname{Var}_U X_\sigma(y).
$$

Это сравнение **uniform-order forced path scoring с exact ELBO-mean scoring**,
не автоматическое разложение настоящего on-policy RTB относительно exact
completion loss. В §3.1 даны другие averaging measures.

## 5. Finite masks: mean bias, square penalty и gradient noise отдельно

Для фиксированных x,y,batch, θ,reference,z:

$$
\widehat D=D_E+B_M+\epsilon,\qquad E_M\epsilon=0.
$$

Тогда

$$
E_M[(\widehat D-b)^2]=(\delta_E+B_M)^2+\operatorname{Var}_M(\widehat D).
$$

Если sampling law масок не зависит от обучаемых parameters, gradient можно
перенести под это конечное ожидание. Поэтому variance term может менять
expected gradient, а не только точность измерения loss.
Внешняя популяция y и sampling measure остаются отдельными вопросами.

### 5.1 IID square

Непустая маска: hidden count uniform 1..u, затем uniform subset этого размера.
$a_M$ — average log ratio по скрытым позициям; actor/reference используют
одинаковую маску. В этом случае BM=0 относительно DE, а при K IID draws

$$
E[\widehat\delta_K^2]=\delta_E^2+v_M/K,\qquad
v_M=\operatorname{Var}(a_M).
$$

Рост K одновременно меняет expected squared objective, update noise и cost.
K→∞ даёт exact mean-ELBO square, не exact marginal loss.
Несмещённость score не равна несмещённости его квадрата.

### 5.2 Local comp: bias относительно полного mean-ELBO

При u>1 эта реализация выбирает l uniform 1..u−1 и дополнение маски.
Каждый член имеет count law без l=u. Если
$m_l=E[a_M\mid |M|=l]$, то

$$
E[\widehat D_{\rm comp}]=\frac1{u-1}\sum_{l=1}^{u-1}m_l,\qquad
B_{\rm comp}=\frac{\bar m_{1:u-1}-m_u}{u}.
$$

Полная маска не учитывается. Квадрат несёт и этот mean shift, и noise term
с covariance между связанными масками. Формула v_single/K без covariance
здесь неприменима. При u=1 код использует полный mask дважды.
Факт local bias проверен; приватные boundaries авторов TraFL этим не установлены.

В опубликованном ESPO full/empty levels и веса заданы иначе.
Использование названия complementary не делает схемы одинаковыми.

### 5.3 Product estimators: что именно они убирают

Пусть $\delta_i=a_{M_i}-b$ — complete residuals из независимых масок.

$$
U_{2+2}=\frac{\delta_1+\delta_2}{2}
\frac{\delta_3+\delta_4}{2},\qquad
U_4=\frac16\sum_{i<j}\delta_i\delta_j.
$$

Обе оценки имеют expectation δE², без vM/4.
Они не убирают ELBO gap, decoder mismatch, mask mean bias или все
optimization difficulties. Individual product estimates могут быть отрицательными.
All-pair при том же K=4 имеет меньшую/равную scalar variance, чем one split,
но это не гарантирует better optimizer outcome.

Для зависимых масок unbiased product требует **независимых unbiased units**,
например двух независимо выбранных правильно взвешенных complementary pairs.
Два зависимых члена одной пары не становятся independent units.
Неправильный product имеет дополнительную covariance bias.

### 5.4 Mask variance и conditional-path preservation не эквивалентны

Контрпример с полностью определёнными нормированными heads, без training:
два binary tokens, reference во всех состояниях предсказывает 0 с probability
1/2. Actor после раскрытия любого другого token также предсказывает 0 с
probability 1/2, но в fully masked state его вероятности 0 на позициях 1/2
равны 3/5 и 5/12. Вероятность символа 1 в каждой голове равна единице минус
вероятность символа 0; все головы нормированы.

Для y=00 обе singleton mask ratios равны 0, а full-mask average ratio
$[\log(6/5)+\log(5/6)]/2=0$. Следовательно DE=0 и mask variance=0.
Но actual actor path masses равны 3/20 и 5/48, поэтому

$$
c_\theta(\sigma_{12}\mid 00)=36/61,\qquad
c_\theta(\sigma_{21}\mid 00)=25/61,
$$

вместо reference 1/2,1/2. Path ratios по порядкам различаются.
Значит даже нулевая mask variance при этом y не удостоверяет preservation.
Это математическое свидетельство, не наблюдение нейронного обучения.

В обратную сторону: два согласованных factorized predictors могут иметь
разные positional log-ratios, ненулевую mask variance и нулевую order variance.
Уменьшение mask variance поэтому не имеет универсальной монотонной связи
с conditional KL.

Напротив, при полном общем support по orders нулевая **order variance**
$\operatorname{Var}_U\log(P_\theta/P_0)=0$ при фиксированном y означает
постоянное log-ratio для всех путей к нему. Нормировка conditional laws тогда
даёт cθ=c0. Именно поэтому explicit relative order-variance penalty является
осмысленным отдельным anchor. Малое значение не является без дополнительных
условий равенством conditional KL; непроверенные/zero-support пути тоже нельзя
исключать незаметно.

## 6. Центрирование и нормировка: не одна операция

| Операция | Что держать отдельным |
|---|---|
| Вычесть deterministic prompt baseline b(x) из reward | Может поглощаться prompt normalizer; cancellation actor gradient требует нормированного exact score |
| Вычесть sample group mean reward, включая собственный sample | При exact fresh on-policy score ослабляет reward term на 1−1/G |
| Разделить reward advantage на group sample SD | Случайный scale зависит от sampled rewards; одного постоянного effective β обычно нет |
| Вычесть detached micro-batch mean score ratio | Связывает score residuals разных samples, иногда разных prompts; не learned prompt partition |
| Вычислить analytic scalar отдельно на каждом rollout group | Profiled finite-batch objective, не deployable learned head |
| Выучить prompt-only normalizer | Может lag/ошибаться; для surrogate score меняет и expected actor direction |
| Делить score на u либо loss на u² | Меняет units, gradient scale и relative weighting prompts разной длины |
| Использовать r=logR или r=R | Меняет reward potential, а не только numerical units |

### 6.1 Finite-group reward centering

Для IID y_i~pθ(·|x), fixed x, G>1, exact normalized score:

$$
E[(r_i-\bar r_G)\nabla\log p_\theta(y_i\mid x)]
=(1-1/G)\nabla E_\theta r.
$$

Prompt-only learned z имеет zero contribution в expected actor gradient
exact score. Поэтому exact completion/path square с этим centering даёт
β_eff=β(1−1/G) в соответствующей KL-minus-reward semigradient формуле §3.2.
При G=5 это 0.8β. Вывод предполагает head parameters отдельно от actor,
fresh on-policy sampling и отсутствие differentiation sampling measure.

Local TB и TraFL центрируют reward; DB/SubTB/EntPPO используют raw terminal r.
На одинаковом nominal beta они не автоматически target-matched.
Это не объявление centering «плохим»: это определение действительного update.
Для masked surrogate нельзя автоматически применить normalized score identity,
так что simple rescaling beta не является доказанным полным исправлением.

### 6.2 Score centering, profile scalar и RSPO

Для fixed group минимизатор $\sum_i(D_i-\beta\tilde r_i/u+z)^2$:

$$
z^*=-\overline{D_i-\beta\tilde r_i/u}.
$$

Если $\overline{\tilde r}=0$, это −meanD. Learned z(x) не равен этому
случайному group optimum по определению.

RSPO использует $\hat d_i=d_i-\mathrm{sg}(\bar d_{\cal B})$ и detached
coefficient A_i−λ hatd_i. При одинаковой convention gradients равны
λ times gradient половины квадрата $(\hat d-A/\lambda)^2$.
Это не равенство losses и не proof равенства prompt-dependent objectives.
Zero-sum advantages могут отменять некоторые производные batch mean;
при partial groups, разных scales/lengths или иной выборке это надо проверять.

Std с ddof=1 — квадратный корень sample variance, рассчитанной с correction;
сама sample SD не является в общем несмещённой оценкой population SD.

## 7. Что уже наблюдали в neural runs, и что осталось причинным вопросом

**Setup из actual settings.** DNA L=8, V=4, post-training signal
r=3.5(z_Potts+z_TFBind8). MLP 40→32→32→32, GELU, 3,424 actor parameters;
start/reference models/pretrain/mlp_5/best.npz, pretrained на Potts coefficient 2.
Seeds 1/2; 10,000 online updates, 256 uniform partial prompts × 5 outputs;
uniform hidden count 1..8 и uniform known letters. Uniform-position rollouts,
temperature 1; fresh single update; constant actor/head Adam rates 1e−4/1e−2;
head width 32; paper normalization; RL beta 0.5; var_lambda 0.
Это 12.8 млн conditional completions на run. Reference frozen.

**Population измерения:** empty prompt, uniform decoder; full 65,536 outputs
и 390,625 states. Conditional error
$E_{p_0}\mathrm{KL}(c_0\Vert c_\theta)$, не own-output entropy, не Jaccard
между разными y. Initial terminal и conditional errors равны 0.

| Arm, final update 10,000 | Seed 1: terminal / conditional KL | Seed 2: terminal / conditional KL |
|---|---:|---:|
| Mask square, IID K=32 | 0.912396 / 0.248274 | 0.920641 / 0.249548 |
| Mask square, local comp K=4 | 0.653096 / 0.490323 | 0.685913 / 0.506483 |
| Exact sampled-path RTB-style TB | 1.229396 / 0.040395 | 1.265868 / 0.039792 |

Terminal KL здесь $\mathrm{KL}(p_0\Vert p_\theta)$.
Полные source paths, joint KL и сохранённые settings/CSV даны в
[audit backing §8](FORMULATIONS_V2_AUDIT_DETAILS.md). Ни строка, ни endpoint здесь не интерполированы.
Проверена арифметика CSV/definitions; neural probabilities не пересчитывались
заново в этой side-проверке.

**Установленное наблюдение:** сохранённая exhaustive evaluation показывает
изменение conditional paths во всех шести runs. При этой настройке conditional
error масочных arms больше, чем TB. Следовательно фраза «это эмпирический
вопрос» не означает, что соответствующих опытов не было.

**Не установленное объяснение:** эти runs не изолируют одну variance penalty.
Одновременно отличаются scoring object, mask distribution/count, forward cost,
expected loss и конечные terminal distances. Они не доказывают неизбежный
дрейф любого completion objective, entropy collapse или optimal method ranking.
Контролируемое causal утверждение требует удержать либо отдельно варьировать
эти оси, а не только показать ненулевой KL.

В saved mask diagnostic на 256 own outputs seed-1 actors / 400 redraws:
comp4 variance меньше IID4, но больше IID32; вместе с этим comp меняет mean
score. Поэтому объяснение «один только размер variance term задаёт neural
conditional distortion» этими данными не поддерживается.

## 8. Декодер, support и policy family — ещё две самостоятельные оси

### 8.1 Rollout-score mismatch

Если generation использует D, а score означает pU, одинаковые network weights
не делают batch on-policy относительно скорящейся вероятности pU.
Для сравнения с actual completion ratio можно формально разделить:

$$
\widehat D-D_{\rm comp}^{D}
=\underbrace{(D_{\rm comp}^{U}-D_{\rm comp}^{D})}_{\text{decoder discrepancy}}
+\underbrace{(J_0^U-J_\theta^U)/u}_{\text{ELBO discrepancy}}
+\underbrace{B_M}_{\text{mask-mean bias}}+\epsilon_M.
$$

Это decomposition scores при finite supported ratios, не proof равенства
losses/gradients. И оно начинается с completion objective: утрата joint
constraint между A и B в §1 остаётся отдельным изменением.

У exact decoder scorer support должен быть настоящим.
Local JAX scorer floors zero competitor events to 1e−30, а NumPy evaluator
сохраняет structural zeros. Поэтому их likelihood claims не буквально
тождественны. При reference-zero пути настоящий relative path residual
может быть бесконечным; tiny floor создаёт другой объект, а не обычную
малую additive approximation. Формулировка «KL труден только при почти
детерминированном sampler» слишком узка.

### 8.2 Representation restriction

При uniform-position actor все marginal orders равны U, но ideal tilted target
может иметь неравномерные orders. Exact path objective тогда также не обязан
достичь zero loss/zero KL. Approximation error и representational floor нужно
разделять до объяснения «не дообучилось».

MLP/Transformer sizes могут менять consistent fitting, conditional allocation
и optimizer behavior. Они не меняют fixed external order prior. Learned
position policy — другая ось, не автоматически просто larger backbone.
Raw joint-action softmax и per-position softmax с uniform position adapter
могут задавать разные policy families даже над теми же logits.

## 9. Что утверждать и что проверять дальше

| Утверждение | Статус |
|---|---|
| Ideal reference tilt оставляет c0 при фиксированном y | Exact identity при заданных normalization/support |
| Exact completion loss равен path loss с меньшим MC noise | Неверно в общем: objectives и expected gradients различаются |
| All-mask enumeration восстанавливает exact marginal | Неверно в общем: восстанавливает DE |
| IID mask score unbiased для DE | Верно при объявленном mask law |
| IID mask square unbiased для DE residual square | Неверно: добавляет v/K |
| Independent product unbiased для mean residual square | Верно при действительно независимых units и finite moments |
| Local comp меняет только variance | Неверно: mean также усечён |
| Removing variance обязательно вызывает path drift | Не theorem; empirical effects надо показывать отдельно |
| Conditional paths изменились в показанных neural runs | Наблюдается в сохранённой exact evaluation; scope §7 |
| Эти изменения однозначно вызваны одним variance term | Не идентифицировано |
| Centering всегда плохой/всегда только variance reduction | Ни то ни другое: эффект зависит от конкретной операции и score |
| Same beta означает same target и fair training | Не без units, centering, decoder и optimizer accounting |

Для новых сравнимых experiments сначала объявить:

1. Reference checkpoint/law, actual decoder и terminal utility versus log potential.
2. Exact joint цель, exact terminal цель или surrogate fitting — что проверяется.
3. Scoring measure по orders и corruption masks, support и numeric floors.
4. Все centerings, нормировки, learned/profiled scalars и expected update assumptions.
5. Task/reference/backbone, actual samples/steps/cost/schedule; что удерживается неизменным.
6. Terminal target error, same-y conditional KL с несколькими populations,
   conditional/output entropy и support statuses — не только один ratio.
7. Original endpoints и seedwise contrasts; interpolation и missing cohorts отдельно.

Это не требует заранее обучить все варианты. Но без этих объявлений нельзя
приписывать разницу «TraFL против RTB» одной математической замене.
Известные zero-support floor и non-square elbo normalization discrepancies
описаны в audit backing §9 (вторая исправлена в коде в `bd78a2c`); этот документ
не сертифицирует новые runs.

## 10. Источники и границы

Implementation:
[path/decoder scoring](src/rl/losses/trajectory_balance.py),
[mask law](src/rl/losses/masks.py), [estimators](src/rl/losses/estimators.py),
[baselines](src/rl/losses/baselines.py), [caller and actual sampling](src/rl/train.py),
[conditional-KL measurement](src/metrics/paths.py),
[Hellinger](src/metrics/decoders.py).
Source snapshot/provenance — [audit backing §10](FORMULATIONS_V2_AUDIT_DETAILS.md).

Paper contexts:
[TraFL October review, Eq. 4 / Apps. C,D.5](../gflownet-dllm/docs/sources/32236_trafl_review_2026-10-06/paper.md),
[ESPO coupled construction](../gflownet-dllm/docs/sources/mds_extracted/2512.03759_wang_espo_sequence_level_rl_diffusion.md),
[JustGRPO objective/sampler distinction](../gflownet-dllm/docs/sources/extracted/2601.15165/tex/5-justGRPO.tex),
[RSPO detached quadratic relation](https://arxiv.org/html/2605.10218v1).

Это новый checked conceptual reference и чтение сохранённых finite-task данных,
не полный audit авторских private implementations, не rerun всех tensor
artifacts, не benchmark verdict и не запуск продолжения main thread.
