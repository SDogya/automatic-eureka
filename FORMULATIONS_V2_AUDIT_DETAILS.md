# Подробности аудита формулировок: исходная расширенная v2

Это сохранённая подробная версия, не основной документ для переноса методов.
Для передачи коллегам начните с [FORMULATIONS_V2.md](FORMULATIONS_V2.md);
для разбора смены objectives — с [TRAFL_OBJECTIVE_DISAMBIGUATION.md](TRAFL_OBJECTIVE_DISAMBIGUATION.md).
Формулы, source snapshot и шесть empirical cases ниже сохранены как backing.

Проверка 2026-10-10 по снимку `1b355c5` (§10); проверявшийся тогда FORMULATIONS.md
снят (история: `e9e4375`). Пункты 2–3 §9 после проверки исправлены в коде (`bd78a2c`). Это справочник по текущему локальному порту,
а не заявление о точном воспроизведении авторских экспериментальных систем.

Начните с §1–3 для определения объектов; §4–6 описывают оценки и обновления;
§7 — метрики; §8 — реально наблюдавшееся изменение условных путей.
Отдельная [записка о переходах между целями и приближениями](TRAFL_OBJECTIVE_DISAMBIGUATION.md)
подробно разбирает, что именно меняется при замене траектории завершением,
ELBO, случайными масками и центрированием. Она не предполагает, что вы уже
приняли интерпретацию экспериментов.

**Статус проверки.** Прочитаны формулировки, соответствующие функции потерь,
вызывающий код, определения метрик и тесты как спецификации. Сохранённые
настройки и CSV шести нейронных запусков в §8 прочитаны непосредственно;
проверена арифметика разложения KL. Малые конечные контрпримеры и эффект
центрирования проверены независимо без обучения. Полный набор JAX-тестов,
повторное нейронное вычисление вероятностей и полное сравнение с авторским
кодом в этой проверке **не выполнялись**. Доступность данных не равна их
независимой повторной генерации.

## 1. Обозначения и конкретная среда

| Объект | Определение |
|---|---|
| $x$ | Начальная частичная строка, которую считаем фиксированным промптом |
| $U(x)$, $u$ | Изначально скрытые позиции и их число; формулы с $1/u$ требуют $u>0$ |
| $y$ | Полное завершение, совпадающее с $x$ на изначально открытых позициях |
| $s_t$ | Частичная строка перед шагом $t$; не индекс случайного порядка |
| $\sigma=(i_1,\ldots,i_u)$ | Перестановка раскрываемых позиций |
| $\tau_\sigma(y)$ | Вся траектория, раскрывающая одно и то же $y$ в порядке $\sigma$ |
| $f_\theta(v\mid s,i)$ | Вероятность буквы из головы маскированного предиктора |
| $\mathcal D$ | Правило генерации: выбор позиции, температура, выбор буквы и прочие решения декодера |
| $P_\theta^\mathcal D(\tau\mid x)$ | Нормированный закон траекторий под объявленным декодером |
| $p_\theta^\mathcal D(y\mid x)$ | Сумма этого закона по траекториям к $y$ |
| $c_\theta^\mathcal D(\sigma\mid x,y)$ | Условный закон порядков при точно том же завершении |
| $r(y)$ | Скалярный сигнал, который действительно передан в функцию потерь |
| $R(y)$ | Положительный потенциал, когда выбрана конвенция $r=\log R$ |
| $\beta$ | Коэффициент в идеальном экстенсивном наклоне $\exp(\beta r)$ |
| $z(x)$ | Обучаемый аддитивный параметр невязки; его единицы нужно объявлять |
| $b(x,y)$ | Скалярный сдвиг невязки, фиксированный относительно выбора порядка/маски |

Для текущей реализации [config.py](src/config.py): $L=8$, алфавит A/C/G/T,
MASK имеет ID 4. Всего $4^8=65\,536$ завершений и $5^8=390\,625$ частичных
строк. При пустом промпте $u=8$, возможны $8!=40\,320$ порядка и $2^8-1=255$
непустых масок. Общие формулы ниже не означают, что этот код автоматически
параметризован произвольными $L,V$: в некоторых функциях они глобальные.

Архитектуру выбирает [architecture.py](src/architecture.py): MLP либо
двунаправленный Transformer. Декодер не превращается в AR только потому,
что одна и та же сеть используется слева направо.

### 1.1 Награда, потенциал и метрика качества — разные объекты

В локальных запусках $r$ берётся из поля log_reward. Для смешанной задачи
[data.py](src/data.py) задаёт

$$
r(y)=3.5\,[z_{\mathrm{Potts}}(y)+z_{\mathrm{TFBind8}}(y)],
\qquad R(y)=\exp r(y),
$$

где обе стандартизации вычисляются по всему пространству строк. Поэтому
наклон равен $R^\beta$, а не $\exp(\beta R)$.
Исходный TFBind8 binding score в $[0,1]$ является отдельной отчётной метрикой:
см. [происхождение данных](data/tfbind8/SOURCE.md). Это не бинарная корректность.

В других задачах, включая RLVR, допустимо передавать непосредственно
скалярную utility $r\in\{0,1\}$. Тогда $\exp(\beta r)$ — нормальный наклон;
логарифмировать такой бинарный сигнал нельзя. Конвенция $r=\log R$ здесь
явная, а не универсальное требование к TraFL.

Коэффициент Поттса при претрейне, масштаб смешанного r=3.5, RL-коэффициент
$\beta$, температура выборки учителя и температура декодера — разные параметры.

## 2. Законы генерации: всегда указывать декодер

### 2.1 Равномерный выбор позиции

Пусть каждый шаг сначала равномерно выбирает скрытую позицию, затем букву
из $f_\theta$ при температуре 1. Тогда

$$
P_\theta^U(\tau_\sigma(y)\mid x)
=\prod_{t=1}^{u}\frac{1}{u-t+1}
 f_\theta(y_{i_t}\mid s_t,i_t).
$$

Общий множитель выбора порядка равен $1/u!$. Поскольку вероятность выбора
позиций не зависит от сети, $P_\theta^U(\sigma\mid x)=1/u!$ для любой $\theta$.
Но это **не** фиксирует $c_\theta^U(\sigma\mid x,y)$.

Если все головы f являются согласованными условными одной положительной
совместной меры $\pi$, то
$P_\theta^U(\tau_\sigma(y)\mid x)=\pi(y\mid x)/u!$ и условный порядок равномерен.
Произвольный обученный маскированный предиктор может быть несогласованным.
Это утверждение о равномерном декодере, не о confidence-декодере.

### 2.2 AR

Позиции выбираются по возрастанию среди $U(x)$; изначально известные символы
промпта сохраняются. Тогда $p_\theta^{AR}$ — нормированное произведение
вероятностей букв по этому порядку. Условный порядок детерминирован:
его энтропия 0 задана декодером, а не доказывает обучение коллапсу путей.

### 2.3 Декодер со случайными предложениями и выбором по confidence

В рассматриваемой локальной адаптации каждая скрытая позиция предлагает
букву из $\mathrm{softmax}(\ell_j/T)$; фиксируется предложение с наибольшей
вероятностью по нетемперированной голове. При равенстве выигрывает меньший индекс.

Нормированный шаговый закон включает вероятность победы над всеми
отброшенными предложениями:

$$
P_\theta^D(i,v\mid s)
=f_\theta^T(v\mid s,i)
\prod_{\substack{j\ \mathrm{hidden}\\j\ne i}}
\Pr\{\text{предложение }j\text{ не побеждает }(i,v)\}.
$$

Нельзя заменить его на $f_\theta(v\mid s,i)$ или $f_\theta(v\mid s,i)/u_t$.
Здесь $\tau$ фиксирует принятые действия; отброшенные предложения
маргинализованы в шаговом законе. При одинаковом y возможны несколько порядков.
Структурные нули возможны даже при $T>0$.

Источники: [семплер](src/rl/samplers.py), [оценка шагового закона и Hellinger](src/metrics/decoders.py),
[дифференцируемый скорер](src/rl/losses/trajectory_balance.py).
Слово «LLaDA/Fast-dLLM» здесь обозначает исследуемую serial-адаптацию;
параллельные/blockwise авторские pipelines этим не воспроизведены.
Оговорка о floor в дифференцируемом скорере дана в §9.

## 3. Идеальные цели — не таблица гарантированных fixed points кода

При нормируемом потенциале и заданном референсном законе:

$$
Q_\beta^\mathcal D(\tau\mid x)
=\frac{P_{\mathrm{ref}}^\mathcal D(\tau\mid x)e^{\beta r(y)}}{Z_\beta^\mathcal D(x)},
\qquad
q_\beta^\mathcal D(y\mid x)
=\frac{p_{\mathrm{ref}}^\mathcal D(y\mid x)e^{\beta r(y)}}{Z_\beta^\mathcal D(x)}.
$$

По определению $Q_\beta^\mathcal D(\sigma\mid x,y)=c_{\mathrm{ref}}^\mathcal D(\sigma\mid x,y)$.
Это свойство **идеальной цели**, не доказательство того, что практический
masked-score residual сохраняет эти условные законы.

| Идеальный объект | Что требуется | Чего из него не следует |
|---|---|---|
| Совместное reference-tilted matching | Точные вероятности одного декодера и правильный log potential | Достижимость Q ограниченной сетью/декодером и сходимость оптимизатора |
| Только terminal matching к q | Точное выполнение отношения terminal probabilities | Сохранение c при том же y |
| Канонический GFlowNet с фиксированным обратным законом | Объявленные terminal reward и $P_B$ | Эквивалентность RTB с pretrained forward reference без проверки факторизации |
| Неанкерированное увеличение utility | Заданный reward-update | Совпадение с q либо обязательный коллапс условных путей |

При uniform backward reveal policy целевой условный закон порядков равномерен.
Чтобы получить именно Q, можно использовать обратный закон, индуцированный
референсной траекторной мерой, и terminal reward $p_{\mathrm{ref}}(y\mid x)R^\beta$.
Этот $P_B$ не равен forward prediction head и при разных x может зависеть от x.

**Достижимость.** Для fixed-uniform-position actor все marginal orders равномерны,
а у Q они могут измениться после наклона:
$Q(\sigma\mid x)=\sum_y q(y\mid x)c_{\mathrm{ref}}(\sigma\mid x,y)$.
Если они неравномерны, точного равенства $P_\theta^U=Q$ в этой семье быть не может.
Большая сеть сама по себе не меняет внешнее правило выбора позиции.

Если фактический terminal reference точно равен $R^{1/T_{\rm teacher}}/Z$,
наклон даёт $R^{1/T_{\rm teacher}+\beta}$, то есть известное «охлаждение».
Приближённо обученный reference или другой декодер не дают этого равенства
автоматически. Та же награда у учителя допустима для механистического контроля;
это не тест переноса на новую награду, но не бессодержательный эксперимент.

## 4. Точные и приближённые лог-отношения

Для uniform decoder, фиксированных x,y:

$$
D_{\rm path}(\sigma,y)=\frac1u\log\frac{P_\theta^U(\tau_\sigma(y)\mid x)}
 {P_{\mathrm{ref}}^U(\tau_\sigma(y)\mid x)},\qquad
D_{\rm comp}(y)=\frac1u\log\frac{p_\theta^U(y\mid x)}{p_{\mathrm{ref}}^U(y\mid x)}.
$$

Для непустой маски $M\subseteq U(x)$:

$$
a_M(y)=\frac1{|M|}\sum_{i\in M}
\log\frac{f_\theta(y_i\mid y_{\neg M},i)}
 {f_{\mathrm{ref}}(y_i\mid y_{\neg M},i)},\qquad
D_{\rm mask}=\sum_{M\ne\emptyset}\frac{a_M}{u\binom{u}{|M|}}.
$$

Условие маски включает исходный промпт. Одна и та же маска используется для
actor и reference. При uniform count $l\in\{1,\ldots,u\}$ и uniform subset
$D_{\rm mask}=E_Ma_M=E_UD_{\rm path}$.

Обозначив $J_\theta(y)=\mathrm{KL}(U\Vert c_\theta^U(\cdot\mid x,y))$:

$$
D_{\rm mask}=D_{\rm comp}+\frac{J_{\rm ref}-J_\theta}{u}.
$$

Каждый ELBO является нижней границей своего log marginal, но **разность ELBO**
не является общей верхней/нижней границей log-ratio. Для другого декодера
$p_\theta^D$ это вообще не идентичность без дополнительного обоснования.

Точный uniform log marginal можно считать по revealed subsets A:

$$
F(\emptyset)=0,\qquad
F(A)=\operatorname{LSE}_{i\in A}
\left[F(A\setminus\{i\})+\log f_\theta(y_i\mid x,y_{A\setminus\{i\}},i)
-\log(u-|A|+1)\right],
$$

и $\log p_\theta^U(y\mid x)=F(U(x))$.
См. [estimators.py](src/rl/losses/estimators.py).
Для confidence decoder вместо $f/u_t$ нужен его полный шаговый закон.

### 4.1 Схемы масок

- IID: K независимых count-uniform/subset-uniform масок; несмещённо для D_mask,
  не для D_comp. При u=8 полный набор содержит 255 масок.
- Local comp: K/2 независимых первых масок с $l\in\{1,\ldots,u-1\}$ и их
  дополнения. При u>1 оценивает другое среднее, без fully masked level.
  При u=1 реализация дублирует единственную маску. K должен быть положительным
  и чётным. Это проверенное свойство локального кода, не закрытая атрибуция
  приватной реализации TraFL.
- ESPO published coupled construction: допускает $l=0,\ldots,u$, задаёт
  нулевой вклад пустой маски и вес $(u+1)/l$ для суммы логарифмов непустой.
  Для intensive score нужно ещё деление на u. Среднее независимых таких
  coupled units нельзя путать с независимостью двух масок внутри пары.
- Общий $t\sim U[0,1]$ с Bernoulli(t) по позициям даёт uniform count
  на 0..u. После **совместного** redraw t и маски при пустой маске получается
  uniform count на 1..u. Redraw только маски при фиксированном t даёт иной закон.

Источники: [masks.py](src/rl/losses/masks.py),
[ESPO, приложение Variance Reduction Strategies](../gflownet-dllm/docs/sources/mds_extracted/2512.03759_wang_espo_sequence_level_rl_diffusion.md),
[RSPO, Appendix B.1](https://arxiv.org/html/2605.10218v1#A2.SS1).
RSPO описывает обработку пустых масок, но это само по себе не устанавливает
точное совпадение локального распределения масок с авторским.

## 5. Невязки и квадратичные оценки

Идеальная intensive residual без group-centering:

$$
\delta_D=D-\beta r/u+z(x).
$$

При exact target z соответствует $\log Z/u$; при extensive path score
используются $\Delta\log P-\beta r+z_{\rm ext}$ и $\log Z$.
Одинаковое имя параметра log_z в коде не означает одинаковые единицы.
Масочный a_M уже intensive: повторное деление на u неправильно.

С фиксированными x,y, batch, parameters и z пусть
$b=\beta\tilde r/u-z$, $\mu=E_M(a_M-b)$ и $v=\operatorname{Var}_M(a_M-b)$.

| Оценка из независимых одинаково распределённых масок | Ожидаемое значение при фиксированном y |
|---|---|
| Square: $(K^{-1}\sum_k\delta_k)^2$ | $\mu^2+v/K$ |
| Split, K=4: $(\delta_1+\delta_2)(\delta_3+\delta_4)/4$ | $\mu^2$ |
| Pairwise, K≥2: $[ (\sum_k\delta_k)^2-\sum_k\delta_k^2]/[K(K-1)]$ | $\mu^2$ |
| Exact masks | $\mu^2$ для count-uniform среднего |
| Exact likelihood | Квадрат другой невязки, с D_comp |

Для K=4 pairwise — среднее трёх balanced splits. При конечных вторых моментах
его scalar variance не больше, чем у одного split:
$\mu^2v+v^2/6$ против $\mu^2v+v^2/4$.
Это не гарантирует лучший Adam endpoint. Отрицательная U-оценка допустима.
Взаимозависимые comp masks нельзя подставлять в формулу несмещённости как
независимые; локальный код это запрещает.

Если оценка score имеет смещение B относительно объявленного среднего,
квадрат оценивает $(\delta+B)^2+\operatorname{Var}(\widehat D)$.
Смена IID на comp может изменить оба слагаемых. Масочная variance не равна
variance по порядкам и не является conditional KL.

Тождество по порядкам
$E_\nu[(D_{\rm path}-b)^2]=(E_\nu D_{\rm path}-b)^2+\operatorname{Var}_\nu D_{\rm path}$
верно для любого заданного $\nu$. Но uniform $\nu=U$ и on-policy conditional
$\nu=c_\theta$ дают разные средние. Подробный вывод — в отдельной записке.

## 6. Что именно делает локальный код

Все formulas ниже относятся к неизменённому снимку кода, а не к заявленным
глобальным оптимумам авторских методов. Основной источник: [train.py](src/rl/train.py).
Роллауты переданы в дифференцирование как данные; их закон не дифференцируется.

| Arm | Реализованный сигнал | Важная граница |
|---|---|---|
| trafl square | Квадрат mask-average ratio минус $(\beta/u)(r-\bar r_G)$ плюс prompt head | Mean-score gap, mask variance и выбранная mask bias; не exact marginal target |
| trafl split / pairwise | U-оценка того же mean-score square, при IID | Убирает ожидаемый mask-variance term, но не ELBO gap и не stochastic gradient noise |
| trafl exact_masks | Полное count-weighted среднее перед квадратом | Exact scoring of surrogate, не exact likelihood |
| trafl exact_lik | Exact uniform-decoder marginal ratio/u перед квадратом | Только terminal constraint; finite-group centering остаётся |
| tb | Exact uniform sampled-path ratio, extensive square, group-centred reward | RTB-style relative loss; не reference-free canonical TB и не неизменённая raw-reward цель |
| tb_dec | Decoder-specific path ratio, extensive square, group-centred reward | В коде zero-event floor; точность/support ограничены §9 |
| db / subtb | Relative step/span residuals с terminal $V(y)=\beta r(y)$ | Raw reward, не group-centred; DB усредняет по всем шагам, SubTB сначала внутри каждой траектории |
| entppo | Soft return $\log P_{\rm ref}-\log\pi_{\rm old}$ на reveal steps и $\beta r$ на exit; PPO/GAE/value loss | Raw reward; exact KL-gradient identity требует matched old/current и GAE λ=1 либо отдельного анализа critic |
| espo / espo_ppo | ELBO-score ratio + clipped advantage update + $\kappa[u(s_\theta-s_{\rm ref})]^2/2$ | k2 — не exact KL value; score/mask/normalization/advantage меняют интерпретацию |
| grpo | $-A\log P_\theta^U(\tau\mid x)$ | Нет fixed reference target; A group-standardized, а не просто raw r |
| justgrpo | AR token PPO, per-sequence divisor u, normalized advantage | AR law при training; иные evaluation decoders — transfer tests |
| rspo | $-\overline{\mathrm{sg}(A-\lambda\hat d)\hat d}$, $\hat d=d-\mathrm{sg}(\bar d_{\cal B})$ | Score centering по micro-batch, advantage — по prompt group; не learned prompt partition |
| var_lambda | $+\lambda_{\rm var}(D_{\rm path}(\sigma_1)-D_{\rm path}(\sigma_2))^2/2$, independent uniform orders | Mean penalty равен order variance; это не измерение mask variance или exact conditional KL |

### 6.0 Формулы остальных локальных потерь

Group advantage для G>1:

$$
\bar r_g=G^{-1}\sum_i r_i,\qquad
s_g=\sqrt{\frac1{G-1}\sum_i(r_i-\bar r_g)^2},\qquad
A_i=\mathrm{sg}\left(\frac{r_i-\bar r_g}{s_g+10^{-4}}\right).
$$

ddof=1 делает sample variance несмещённой, но не её квадратный корень s_g.
Определим $C_\epsilon(\rho,A)=\min(\rho A,\mathrm{clip}(\rho,1-\epsilon,1+\epsilon)A)$.
Батч усредняется по valid sequences (u>0); для default prompts они все valid.

Для GRPO локальная потеря равна
$-\overline{A_i\log P_\theta^U(\tau_i\mid x_i)}$.
Для JustGRPO:

$$
\rho_{it}=\exp[\ell_{\theta,it}^{AR}-\mathrm{sg}(\ell_{{old},it}^{AR})],
\qquad
\mathcal L_{\rm JustGRPO}
=-\overline{\frac1{u_i}\sum_{t=1}^{u_i}C_\epsilon(\rho_{it},A_i)}.
$$

При single fresh update old совпадает с текущим actor в начальной точке:
ratio численно 1, но его производная не нулевая. Локальная loss aggregation
не заявляется совпадающей с каждой boundary/zero-advantage деталью авторского
репозитория. AR constraint применяется к rollout и scoring, не к attention mask.

Для ESPO пусть sθ — mean per-token masked log score, а одинаковые mask draws
использованы для current/old/reference:

$$
\rho_i=\exp[s_{\theta,i}-\mathrm{sg}(s_{{old},i})],\qquad
\mathcal L_{\rm ESPO}
=\overline{-C_\epsilon(\rho_i,A_i)
+\frac\kappa2\,[u_i(s_{\theta,i}-s_{{ref},i})]^2}.
$$

Single update использует detached текущий score как old; multi-epoch вариант
держит old policy фиксированным на повторно используемом batch. Mask-score
ratio не объявляется точным importance ratio actual completion law.

Для relative DB/SubTB обозначим
$d_t=\log P_\theta(a_t\mid s_t)-\log P_0(a_t\mid s_t)$.
Конечное $V(s_u)=\beta r(y)$ фиксируется, остальные V обучаются:

$$
\eta_t=V(s_t)+d_t-V(s_{t+1}),\qquad
\eta_{ij}=V(s_i)+\sum_{t=i}^{j-1}d_t-V(s_j).
$$

DB — сумма $\eta_t^2$ по всем valid steps, делённая на их общее число.
SubTB — average $\eta_{ij}^2$ с весами $\lambda_{\rm span}^{j-i}$,
нормированными внутри каждой траектории, затем mean по траекториям.
Сумма residuals телескопируется; **сумма квадратов не равна квадрату суммы**.
При разных u DB и per-trajectory limit SubTB дополнительно различаются
aggregation weights.

Для локального EntPPO reveal return
$g_t=\log P_0(a_t\mid s_t)-\log\pi_{\rm old}(a_t\mid s_t)$,
exit return $g_u=\beta r(y)$, sink value 0. Обозначим
$e_t=g_t+V_{\rm old}(s_{t+1})-V_{\rm old}(s_t)$ и
$A_t^{GAE}=\mathrm{sg}(\sum_{k=t}^{u}\lambda_{\rm GAE}^{k-t}e_k)$.
Actor loss — отрицательное среднее по траекториям суммы

$$
\sum_{t<u}\left[
C_\epsilon(\rho_t,A_t^{GAE})
-\mathrm{KL}(\pi_\theta(\cdot\mid s_t)\Vert\pi_{\rm old}(\cdot\mid s_t))
\right]+A_u^{GAE}.
$$

Здесь $\rho_t=\exp[\mathrm{clip}(\ell_{\theta,t}-\ell_{{old},t},-80,80)]$
в текущем коде, а policy KL охватывает весь legal action distribution
с fixed uniform position rule. Exit actor term detached; derivative 0.
Value loss — взвешенная MSE по valid states к detached
$A_t^{GAE}+V_{\rm old}(s_t)$. В отличие от DB, terminal V(y) здесь обучается,
а не напрямую фиксируется равным βr.
Для λ_GAE<1 и неточного critic это не та же самая оценка full-return gradient.

Для RSPO при $\lambda>0$ и том же detached centering:

$$
\nabla\mathcal L_{\rm RSPO}
=\lambda\nabla\frac12\,\overline{(\hat d-A/\lambda)^2}.
$$

Это равенство gradients, не forward loss values. Для unstandardized A и
одинакового u простое сопоставление intensive reward-score targets даёт
$\beta_{\rm ext}/u=1/\lambda$, а не $\beta_{\rm ext}=1/\lambda$.
Случайная group SD и неодинаковые u не дают одного общего соответствия.
При $\lambda=0$ допустима advantage-weighted форма, но цель A/λ не определена.

### 6.1 Centering и expected semigradient

Пусть score является **точным нормированным** log probability того же закона,
из которого получена fresh IID group; z prompt-only, независим от actor.
Без group centering expected intensive actor semigradient равен

$$
g_{\rm comp}=\frac{2}{u^2}\nabla_\theta
[\mathrm{KL}(p_\theta\Vert p_{\rm ref})-\beta E_{p_\theta}r].
$$

Для path score замените p на P. Это expected update field, а не полный
градиент $E_{p_\theta}\delta^2$: последний включает производную sampling measure.

Если reward baseline — group mean, включая собственный sample:

$$
E[(r_i-\bar r_G)\nabla\log p_\theta(y_i)]
=(1-1/G)\nabla E_{p_\theta}r.
$$

Следовательно в скобках вместо β стоит β(1−1/G). При G=5 фактор 4/5.
В §6 показано, какие arms центрируют reward, а какие используют raw reward.
Стандартизированные advantages, surrogate scores, повторные PPO epochs,
общие actor/head параметры и profiled batch scalars требуют своего вывода;
этот результат нельзя переносить на них автоматически.

В частности, для masked score обычно нет score-function identity
$E_{p_\theta}\nabla s_\theta=0$. Поэтому learned/fixed normalization может
менять ожидаемое actor direction, а не только variance. При exact likelihood
prompt-only baseline отменяется в expected actor gradient; finite-batch noise
и nonlinear optimizer dynamics всё равно могут зависеть от него.

## 7. Метрики и допустимое прочтение

Все KL/энтропии — nats. «Exact» означает суммирование по конечному пространству
с floating-point вычислением голов, не отсутствие численных ошибок.
Промпт и декодер должны совпадать во всех сравниваемых законах.

$$
\mathrm{KL}(P_{\rm ref}\Vert P_\theta)
=\mathrm{KL}(p_{\rm ref}\Vert p_\theta)
E_{p_{\rm ref}}\mathrm{KL}(c_{\rm ref}\Vert c_\theta).
$$

Локальный path KL — последнее слагаемое, с **reference output weighting**.
Это не own-output entropy и не target-weighted KL. Изменение численного
path KL подтверждает изменение условных путей в этой популяции; не говорит,
какие y изменились и является ли изменение коллапсом.
См. [paths.py](src/metrics/paths.py).

Для decoder Hellinger excess:

$$
h^2_{\rm joint}-h^2_{\rm terminal}
=\sum_y\sqrt{p_{\rm ref}(y)p_\theta(y)}
 h^2(c_{\rm ref}(\cdot\mid y),c_\theta(\cdot\mid y)).
$$

Это overlap-weighted conditional distance. При малом terminal overlap малая
excess не удостоверяет сохранение путей. При нулевой массе conditional
distribution не определён: это не entropy 0 и не доказательство коллапса.

Joint entropy раскладывается как $H(Y)+E_{p_\theta}H(\sigma\mid Y)$.
Оба слагаемых нужны; own-weighted conditional entropy меняется и от смеси y.
При fixed uniform marginal orders
$H(\sigma\mid Y)\ge\log u!-H(Y)$: terminal concentration сама может заставить
own-output conditional entropy быть почти максимальной.

Ожидаемое число разных строк в k IID draws:
$\sum_y[1-(1-p(y))^k]$. Для выбранного множества T суммировать только по T.
Top-1% by reward — определение локальной категории, не автоматически
«правильные ответы», смысловые modes или paper Pass@k.

При matched-distance анализе first crossing и log-log interpolation —
прозрачный post-hoc способ сравнения, не наблюдавшаяся intermediate policy.
Одинаковый terminal KL не означает одинаковый terminal law, reward или
оптимальную настройку. Нужны actual endpoints, crossing coverage по seeds и
чувствительность к checkpoint spacing. Рост path/terminal ratio в полосе
terminal KL не доказывает стационарность всего terminal distribution.

## 8. Эмпирический ответ: условные пути действительно изменялись

Это не только открытый теоретический вопрос. Сохранённые нейронные запуски
этой среды дают ненулевые conditional KL. Ниже — факты из settings.json
и evals.csv, не повторный запуск и не интерполяция.

**Setup.** ДНК длины 8; reward r=3.5(z_Potts+z_TFBind8).
Reference/start: [MLP k=5 checkpoint](models/pretrain/mlp_5/best.npz),
архитектура 40→32→32→32, GELU, 3,424 actor parameters.
Претрейн — Potts coefficient 2, не новая fitted teacher в этой проверке.
Для трёх показанных arms: seeds 1/2, 10,000 fresh online updates;
256 random partial prompts × 5 completions, то есть 12.8 млн completions
на case. Initial hidden count uniform 1..8, известные буквы uniform A/C/G/T.
Uniform-position rollout, temperature 1; reference заморожен.
Adam actor LR 0.0001, head LR 0.01, constant schedule, head width 32;
normalization paper, RL β=0.5, variance penalty 0, plateau stop выключен.
Ни transformer, ни AR/confidence training в этих шести cases нет.
Маски: IID K=32 либо local comp K=4; TB использует sampled exact path score.

**Evaluation.** Пустой промпт, uniform-position decoder, весь output/state space.
$K_{\rm term}=\mathrm{KL}(p_{\rm ref}\Vert p_\theta)$,
$K_{\rm cond}=E_{p_{\rm ref}}\mathrm{KL}(c_{\rm ref}\Vert c_\theta)$.
На initial checkpoint оба равны 0.

| Actual trained arm / seed, update 10,000 | $K_{\rm term}$ | $K_{\rm cond}$ | $K_{\rm joint}=K_{\rm term}+K_{\rm cond}$ |
|---|---:|---:|---:|
| TraFL square, IID 32 / 1 | 0.912396 | 0.248274 | 1.160670 |
| TraFL square, IID 32 / 2 | 0.920641 | 0.249548 | 1.170189 |
| TraFL square, local comp 4 / 1 | 0.653096 | 0.490323 | 1.143418 |
| TraFL square, local comp 4 / 2 | 0.685913 | 0.506483 | 1.192396 |
| Sampled-path RTB-style TB / 1 | 1.229396 | 0.040395 | 1.269791 |
| Sampled-path RTB-style TB / 2 | 1.265868 | 0.039792 | 1.305661 |

Прямые backing directories:
[IID32 seed 1](models/port_grid2/trafl_iid32_b0.5/seed_1/finetune_5/settings.json),
[IID32 seed 2](models/port_grid2/trafl_iid32_b0.5/seed_2/finetune_5/settings.json),
[comp4 seed 1](models/port_grid2/trafl_comp4_b0.5/seed_1/finetune_5/settings.json),
[comp4 seed 2](models/port_grid2/trafl_comp4_b0.5/seed_2/finetune_5/settings.json),
[TB seed 1](models/port_grid2/tb_b0.5/seed_1/finetune_5/settings.json),
[TB seed 2](models/port_grid2/tb_b0.5/seed_2/finetune_5/settings.json).
В каждом каталоге есть evals.csv. [SPEC](results/port_grid2/SPEC.md) и
[исходный отчёт](results/report/REPORT.md) сохраняют историю и дополнительные arms.

**Что следует.** По сохранённой exhaustive evaluation conditional paths
отошли от reference в обоих seeds всех трёх arms; при этой настройке
масочные квадраты имеют больший reference-weighted conditional KL, чем TB.
Это наблюдение, а не отсутствие эмпирического ответа.

**Чего не следует.** Это не matched-terminal-distance ranking, не proof
неизбежного дрейфа любой per-string модели и не единственная причина variance.
K, mask mean/bias, expected loss и compute cost различаются; terminal
distances тоже различаются. Ограниченная fixed-position семья, концентрированный
reference, partial-prompt training versus empty-prompt evaluation и два seeds
ограничивают перенос. Ненулевой KL не доказывает entropy collapse.
Эта проверка не удостоверяет каждый endpoint свежим weight-to-law replay.

В [mask_penalty.json](results/report/mask_penalty.json) / [script](scripts/mask_penalty.py)
отдельно измерены 400 mask redraws для 256 собственных outputs двух seed-1
final actors. Comp4/IID4 variance ratios равны 0.509969 / 0.405373;
comp4/IID32 — 4.088834 / 3.239002. Но score means также меняются:
0.095645→0.062801 и 0.029614→0.025569. Это не variance-only intervention.

## 9. Известные реализации/оговорки, которые нельзя спрятать в формулировке

1. **Decoder floor.** Дифференцируемый proposal_step_log_law заменяет нулевые
   competitor CDF на минимум 1e-30. В NumPy evaluation они остаются нулевыми.
   Поэтому слово «exact» для training decoder log law требует оговорки.
   Например, q0=(0.7,0.1,0.1,0.1), q1=(0.97,0.01,0.01,0.01), T=1:
   non-peak proposal позиции 1 не может выиграть, но floor даёт mass 1e-32.
   Reference support/log-ratio semantics меняются; это не только rounding.
2. **Normalization option.** При non-square estimator caller переключает
   reward scaling для normalization=elbo, но estimator продолжает возвращать
   intensive ratio. Для этой комбинации нет заявленного единообразия единиц.
   Показанные §8 runs используют paper и не затронуты этой конкретной веткой.
   **Исправлено в `bd78a2c`:** оценщики, кроме square, всегда используют β/u.
3. **Domain checks.** Pairwise требует K≥2, split — K=4, comp — положительное
   чётное K. Config допускает mask_samples=1; формулу pairwise тогда применять
   нельзя. **Исправлено в `bd78a2c`:** split/pairwise при K<2 и с comp-парами
   вызывают ValueError (тест `test_pairwise_needs_two_masks`).
4. **Границы API.** Default reference=initial означает фактический checkpoint,
   не точную pretraining teacher. Defaults не являются настройками всех runs.
   Legacy README с K=4/model-derived prompts не заменяет settings конкретного case.
5. **Атрибуция.** Сходство выражений с paper objective не доказывает sampler,
   boundary masks, advantage scaling, optimizer или benchmark fidelity.
   Uniform-order coherence/ELBO identities должны иметь явную область применения.

## 10. Источники и воспроизводимый снимок

Снимок проверяемой ветки port-baselines:
HEAD 1b355c5330786ea66fb3940344488074e520b0b6.
Это идентификатор чтения, не новый commit этой записки.

| Файл | SHA-256 прочитанного содержимого |
|---|---|
| FORMULATIONS.md (снят) | c4cf94a51d3e57727b8d5b0454f3baa3ab6450be0e7bf3ea589ae5a1b5a2e795 |
| src/rl/train.py | 70a08a49c72a41dbea5b19d7df1ad291e944a15887ee3120753a743d11cf0e9b |
| src/rl/losses/estimators.py | 15641256f2b03710cd8c38300987152ff4166edce483ea81a5716cd893d3b46b |
| src/rl/losses/trajectory_balance.py | d8f6bb456a50d6c8dc07c4ed96061a6eb97f40cfeea51c763639e946c26b9179 |
| src/rl/losses/baselines.py | daf5a5a4d2010c7dbc162bb34b4f1fa19bc5611bdef6c263cb7d79adb0c96cb4 |

Ключевые implementation references:
[TraFL](src/rl/losses/trafl.py), [mask estimators](src/rl/losses/estimators.py),
[RTB/decoder scoring](src/rl/losses/trajectory_balance.py),
[DB/SubTB](src/rl/losses/flow_balance.py), [EntPPO](src/rl/losses/entppo.py),
[PG/ESPO/RSPO](src/rl/losses/baselines.py),
[matched comparisons](src/analysis/compare.py).

Первичные тексты:
[TraFL October review extraction, Eq. 4 / Apps. A,C,D.5](../gflownet-dllm/docs/sources/32236_trafl_review_2026-10-06/paper.md),
[JustGRPO formulation/training versus inference](../gflownet-dllm/docs/sources/extracted/2601.15165/tex/5-justGRPO.tex),
[ESPO](../gflownet-dllm/docs/sources/mds_extracted/2512.03759_wang_espo_sequence_level_rl_diffusion.md),
[RSPO v1, §§3/B/D](https://arxiv.org/html/2605.10218v1).
Проверены относящиеся к этим формулам разделы, не полный literature audit.
