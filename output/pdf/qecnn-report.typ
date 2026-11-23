#set document(
  title: "Восстановление изображений после сжатия x265",
  author: "Синицин Иван Александрович",
)
#set page(width: 960pt, height: 540pt, margin: (x: 52pt, top: 40pt, bottom: 36pt))
#set text(font: "Segoe UI", size: 20pt, lang: "ru")
#set par(leading: 0.54em)

#let ink = rgb("#17202a")
#let muted = rgb("#536171")
#let grid-color = rgb("#dce2e7")
#let axis-color = rgb("#9ca9b4")
#let codec-color = rgb("#7a828d")
#let base-color = rgb("#5260a9")
#let model-color = rgb("#06766a")
#set text(fill: ink)

#let results = json("../../results.json")
#let scores = results.per_image_psnr_db
#let means = results.mean_psnr_db
#let codec = scores.at("x265")
#let baseline = scores.at("QECNN")
#let restored = scores.at("QECNN improved")
#assert((codec, baseline, restored).all(values => values.len() == 100))
#let differences = restored.zip(baseline).map(pair => pair.first() - pair.last())
#let gain = results.gain_over_baseline_db
#let improved-frames = differences.filter(value => value > 0).len()

#let decimal(value, digits: 3) = {
  let pieces = str(calc.round(calc.abs(value), digits: digits)).split(".")
  let fraction = pieces.at(1, default: "")
  let sign = if value < 0 { "−" } else { "" }
  sign + pieces.first() + "," + fraction + "0" * (digits - fraction.len())
}

#let small(body) = text(size: 14pt, fill: muted, body)
#let title(body) = [
  #text(size: 35pt, weight: "semibold", body)
  #v(15pt)
]

#let label(x, y, body) = place(top + left, dx: x, dy: y, small(body))
#let plot(series, low, high, ticks, height, show-x: false, mean: none) = {
  let chart-x = 52pt
  let chart-width = 774pt
  let y(value) = height * (high - value) / (high - low)
  let x(index) = chart-width * index / 99
  block(width: 100%, height: height + if show-x { 22pt } else { 0pt })[
    #for (value, caption) in ticks {
      place(top + left, dx: chart-x, dy: y(value), line(length: chart-width, stroke: 0.5pt + grid-color))
      label(1pt, y(value) - 8pt, box(width: 40pt, align(right, caption)))
    }
    #place(top + left, dx: chart-x, line(end: (0pt, height), stroke: 0.65pt + axis-color))
    #place(top + left, dx: chart-x, dy: height, line(length: chart-width, stroke: 0.65pt + axis-color))
    #if mean != none {
      place(top + left, dx: chart-x, dy: y(mean), line(
        length: chart-width, stroke: (paint: base-color, thickness: 0.85pt, dash: "dashed"),
      ))
    }
    #for (values, color, thickness) in series {
      let points = values.enumerate().map(((index, value)) => (x(index), y(value)))
      let segments = points.slice(1).map(point => curve.line(point))
      place(top + left, dx: chart-x, curve(
        stroke: thickness + color, curve.move(points.first()), ..segments,
      ))
    }
    #if show-x {
      for frame in (1, 20, 40, 60, 80, 100) {
        let marker = chart-x + x(frame - 1)
        place(top + left, dx: marker, dy: height, line(end: (0pt, 4pt), stroke: 0.65pt + axis-color))
        label(marker - 19pt, height + 6pt, box(width: 38pt, align(center, str(frame))))
      }
    }
  ]
}

#let legend(name, value, color) = [
  #box(line(length: 22pt, stroke: 1.7pt + color))
  #h(5pt)
  #text(size: 16pt, weight: "medium", name)
  #h(3pt)
  #small[#decimal(value) дБ]
]

#v(50pt)
#small[Лабораторная работа по теории информации]

#v(24pt)
#text(size: 48pt, weight: "semibold")[
  Восстановление изображений \
  после сжатия x265
]

#v(16pt)
#text(size: 24pt, fill: muted)[Модификация QECNN на BSD500]

#v(58pt)
#text(size: 22pt, weight: "semibold")[Синицин Иван Александрович] \
#text(size: 19pt, fill: muted)[Группа М4145]

#pagebreak()
#title[Задача и данные]

В задании нужно получить *не менее +0,10 дБ* в среднем по 100 тестовым изображениям относительно базовой QECNN.

#v(22pt)
#table(
  columns: (1fr, 1fr, 3fr),
  inset: (x: 10pt, y: 12pt),
  stroke: (x: none, y: 0.6pt + grid-color),
  table.header([*Набор*], [*Кадры*], [*Назначение*]),
  [Обучающий], [200], [160 для обучения и 40 для валидации. После выбора настроек используются все 200.],
  [Тестовый], [100], [Оценка на целых кадрах 480×320.],
)
#v(23pt)
Для каждого изображения есть оригинал и результат декодирования x265. Сеть получает декодированный кадр и предсказывает поправку к RGB. Битовый поток x265 остаётся прежним.

#pagebreak()
#title[Что изменилось в QECNN]

Перед выходной свёрткой я добавил остаточный блок: две свёртки 3×3 с PReLU между ними. Выход блока прибавляется к признакам исходной сети.

#v(14pt)
#table(
  columns: (2fr, 1fr, 1fr, 1.2fr),
  inset: (x: 10pt, y: 9pt),
  align: (left, center, center, right),
  stroke: (x: none, y: 0.55pt + grid-color),
  table.header([*Модель*], [*Шаг ядер*], [*Поле зрения*], [*Параметры*]),
  [Базовая QECNN], [—], [21×21], [1 389 123],
  [Контрольная], [1 и 1], [25×25], [1 463 043],
  [Модифицированная], [2 и 4], [33×33], [1 463 043],
)
#v(12pt)
При квадратичной ошибке оптимальная поправка к пикселю — средняя ошибка декодирования при известном окружении. Более широкое поле зрения даёт сети больше соседних пикселей для этой оценки.

#v(8pt)
#small[Первый вариант с шагами 2 и 4 набрал +0,020 дБ. Контрольная модель дала +0,026 дБ. Итоговый выигрыш измерен после дообучения и улучшения инференса.]

#pagebreak()
#title[Как обучалась итоговая модель]

#table(
  columns: (3.4fr, 1fr, 1.1fr, 0.9fr),
  inset: (x: 10pt, y: 10pt),
  align: (left, center, center, center),
  stroke: (x: none, y: 0.6pt + grid-color),
  table.header([*Этап*], [*Кадры*], [*Фрагмент*], [*Эпохи*]),
  [Обучение с нуля], [160], [64×64], [25],
  [Дообучение с выбором весов по валидации], [160], [64×64], [8],
  [Дообучение на всех обучающих кадрах], [200], [64×64], [8],
  [Последний этап на крупных фрагментах], [200], [96×96], [6],
)

#v(20pt)
При дообучении оригинал и сжатый фрагмент поворачиваются и отражаются одинаково. Кадры с меньшей исходной MSE получают больший вес, поскольку одинаковое снижение MSE даёт им больший прирост в дБ.

#v(12pt)
Ошибку считаю с отступом 16 пикселей от краёв. У фрагмента 96×96 в расчёт попадает центральная область 64×64.

#pagebreak()
#title[Восстановление тестового кадра]

+ Беру четыре поворота кадра, каждый с отражением и без него.
+ Для каждого варианта добавляю по 16 пикселей зеркального окружения. После прохода сети отступ убираю.
+ Возвращаю восемь оценок в исходную ориентацию и усредняю их по пикселям.

#v(26pt)
Зеркальные края уменьшают пограничные артефакты. Усреднение сглаживает различия между оценками. При MSE ошибка среднего ограничена средней ошибкой отдельных оценок на том же кадре.

#v(14pt)
На 40 валидационных кадрах зеркальные края дали *+0,039 дБ* при тех же весах wide.

#v(9pt)
#small[На один кадр приходится восемь проходов сети. #link("https://openaccess.thecvf.com/content_cvpr_2017_workshops/w12/papers/Lim_Enhanced_Deep_Residual_CVPR_2017_paper.pdf")[Геометрическое усреднение в EDSR, раздел 4.3.]]

#pagebreak()
#[
#set par(spacing: 0.35em)
#title[PSNR на 100 изображениях]

#grid(
  columns: (1fr, 1.3fr, 1.4fr),
  legend("x265", means.at("x265"), codec-color),
  legend("Базовая QECNN", means.at("QECNN"), base-color),
  legend("Модифицированная QECNN", means.at("QECNN improved"), model-color),
)
#v(9pt)
#small[RGB PSNR, дБ]
#v(5pt)
#plot(
  ((codec, codec-color, 0.95pt), (baseline, base-color, 1pt), (restored, model-color, 1.35pt)),
  18, 35.5, ((18, "18"), (22, "22"), (26, "26"), (30, "30"), (34, "34")), 136pt,
)
#v(9pt)
#small[Прирост к базовой QECNN, дБ]
#v(6pt)
#plot(
  ((differences, model-color, 1.25pt),),
  -0.02, 0.53, ((0, "0"), (0.2, "0,2"), (0.4, "0,4")), 82pt,
  show-x: true, mean: gain,
)
#align(center, small[Номер тестового изображения])
#v(7pt)
#small[Пунктир — средний прирост *+#decimal(gain) дБ*. Улучшены #improved-frames из 100 кадров. Средний выигрыш над x265 — *+#decimal(results.gain_over_x265_db) дБ*.]
]

#pagebreak()
#title[Визуальное сравнение]

#block(width: 100%, height: 380pt)[
  #let image-x = 228pt
  #for (column, caption) in ("Оригинал", "x265", "Базовая QECNN", "Модифицированная").enumerate() {
    place(top + left, dx: image-x - 20pt + column * 156pt, dy: 1pt, box(
      width: 144pt, align(center, text(size: 14pt, weight: "semibold", caption)),
    ))
  }
  #place(top + left, dx: image-x, dy: 27pt, image("examples.png", width: 572pt))
  #for (row, (frame, caption)) in ((96, "Минимум"), (21, "Медиана"), (63, "Максимум")).enumerate() {
    let y = row * 117pt
    let difference = differences.at(frame - 1)
    place(top + left, dx: 4pt, dy: y + 44pt)[
      *Кадр #frame* \
      #small[#caption #if difference >= 0 { "+" }#decimal(difference) дБ]
    ]
  }
]

#pagebreak()
#title[Репозиторий и запуск]

#link("https://github.com/i3373/IMENH")[#text(size: 26pt, fill: model-color)[github.com/i3373/IMENH]]

#v(20pt)
Для запуска на Windows нужен Python 3.11. После установки зависимостей по README батник запускается из папки проекта в PowerShell.

#v(10pt)
#block(fill: rgb("#f0f3f5"), inset: 14pt, width: 100%, raw(
  ".\\train_and_compare.bat --fresh", lang: "powershell",
))

#v(16pt)
Батник обучит модели с нуля и сохранит график, примеры и метрики в #raw("runs/repro-*/final").

#v(9pt)
#small[Файл #raw("best_model2026.pth") из задания нужен для сравнения с базовой QECNN.]

