from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = ROOT / "demo" / "documents"
DATASET = ROOT / "demo" / "benchmark.jsonl"

TOPICS = {
    "en": [
        (
            "cancel-order",
            "Order cancellation",
            "An order can be cancelled before it is handed to the courier.",
        ),
        ("returns", "Returns", "Unused products can be returned within 14 calendar days."),
        ("delivery", "Delivery", "Standard delivery takes two to four business days."),
        (
            "payment",
            "Payment methods",
            "Cards and bank transfers are accepted; cash is not accepted.",
        ),
        ("password", "Password reset", "A password reset link remains valid for 30 minutes."),
        ("subscription", "Subscription", "Subscriptions can be paused for up to three months."),
        ("invoice", "Invoices", "Invoices are generated after payment and appear in Billing."),
        ("support", "Support", "Priority support operates every day from 08:00 to 22:00."),
        (
            "delete-data",
            "Data deletion",
            "Account data is deleted within 30 days after verification.",
        ),
    ],
    "ru": [
        (
            "cancel-order",
            "Отмена заказа",
            "Заказ можно отменить до его передачи курьеру.",
        ),
        (
            "returns",
            "Возврат",
            "Неиспользованный товар можно вернуть в течение 14 календарных дней.",
        ),
        (
            "delivery",
            "Доставка",
            "Стандартная доставка занимает от двух до четырех рабочих дней.",
        ),
        (
            "payment",
            "Способы оплаты",
            "Принимаются карты и банковские переводы; наличные не принимаются.",
        ),
        (
            "password",
            "Сброс пароля",
            "Ссылка для сброса пароля действует 30 минут.",
        ),
        (
            "subscription",
            "Подписка",
            "Подписку можно приостановить на срок до трех месяцев.",
        ),
        (
            "invoice",
            "Счета",
            "Счет создается после оплаты и появляется в разделе Billing.",
        ),
        (
            "support",
            "Поддержка",
            "Приоритетная поддержка работает ежедневно с 08:00 до 22:00.",
        ),
        (
            "delete-data",
            "Удаление данных",
            "Данные аккаунта удаляются в течение 30 дней после проверки.",
        ),
    ],
    "uz": [
        (
            "cancel-order",
            "Buyurtmani bekor qilish",
            "Buyurtma kuryerga topshirilishidan oldin bekor qilinishi mumkin.",
        ),
        (
            "returns",
            "Mahsulotni qaytarish",
            "Ishlatilmagan mahsulotni 14 kalendar kun ichida qaytarish mumkin.",
        ),
        (
            "delivery",
            "Yetkazib berish",
            "Standart yetkazib berish ikki-to'rt ish kuni davom etadi.",
        ),
        (
            "payment",
            "To'lov usullari",
            "Karta va bank o'tkazmasi qabul qilinadi; naqd pul qabul qilinmaydi.",
        ),
        ("password", "Parolni tiklash", "Parolni tiklash havolasi 30 daqiqa amal qiladi."),
        ("subscription", "Obuna", "Obunani uch oygacha to'xtatib turish mumkin."),
        (
            "invoice",
            "Hisob-faktura",
            "Hisob-faktura to'lovdan keyin Billing bo'limida yaratiladi.",
        ),
        ("support", "Yordam xizmati", "Ustuvor yordam har kuni 08:00 dan 22:00 gacha ishlaydi."),
        (
            "delete-data",
            "Ma'lumotlarni o'chirish",
            "Akkaunt ma'lumotlari tekshiruvdan keyin 30 kun ichida o'chiriladi.",
        ),
    ],
}

QUERIES = {
    "en": {
        "cancel-order": [
            "When may I cancel an order?",
            "Can a purchase be called off?",
            "Cancelling orders before shipping",
            "Can I cancel and still ask about delivery?",
            "What does otmena zakaza allow me to do?",
        ],
        "returns": [
            "What is the return period?",
            "How long do I have to send an item back?",
            "Rules for returning unused products",
            "Can I return an item and cancel another order?",
            "What is the srok vozvrata for unused goods?",
        ],
        "delivery": [
            "How long is standard delivery?",
            "When should my parcel arrive?",
            "Normal shipping duration",
            "Tell me delivery time and accepted payment",
            "Yetkazib berish usually takes how long?",
        ],
        "payment": [
            "Which payment methods are accepted?",
            "Can I pay cash?",
            "Ways to pay for an order",
            "Can I pay by card and get an invoice?",
            "Is oplata nalichnymi available?",
        ],
        "password": [
            "How long is a reset link valid?",
            "When does the password link expire?",
            "Password recovery timeout",
            "Reset my password and delete old data",
            "Parolni tiklash link duration",
        ],
        "subscription": [
            "Can I pause my subscription?",
            "How long can the plan be suspended?",
            "Subscription pause limits",
            "Pause a subscription and contact support",
            "Can I priostanovit obunu?",
        ],
        "invoice": [
            "Where can I find my invoice?",
            "When is a bill generated?",
            "Invoice availability after payment",
            "I paid by card; where is the invoice?",
            "Where is hisob-faktura shown?",
        ],
        "support": [
            "What are priority support hours?",
            "When can I contact support?",
            "Priority help schedule",
            "Support hours and subscription pauses",
            "Yordam xizmati working hours",
        ],
        "delete-data": [
            "How quickly is account data deleted?",
            "When will my information be erased?",
            "Account deletion processing time",
            "Delete my data after resetting the password",
            "Ma'lumotlarni o'chirish takes how long?",
        ],
    },
    "ru": {
        "cancel-order": [
            "Когда можно отменить заказ?",
            "Можно ли отказаться от покупки?",
            "Условия отмены заказов",
            "Можно отменить заказ и узнать о доставке?",
            "Как работает order cancellation?",
        ],
        "returns": [
            "Какой срок возврата?",
            "Сколько времени есть на возврат товара?",
            "Правила возврата неиспользованных товаров",
            "Можно вернуть товар и отменить другой заказ?",
            "Какой return period у товара?",
        ],
        "delivery": [
            "Сколько длится стандартная доставка?",
            "Когда приедет посылка?",
            "Срок обычной доставки",
            "Расскажите о доставке и способах оплаты",
            "Сколько занимает yetkazib berish?",
        ],
        "payment": [
            "Какие способы оплаты принимаются?",
            "Можно ли платить наличными?",
            "Чем оплатить заказ?",
            "Можно оплатить картой и получить счет?",
            "Доступен ли cash payment?",
        ],
        "password": [
            "Сколько действует ссылка сброса?",
            "Когда истечет ссылка для пароля?",
            "Срок восстановления пароля",
            "Сбросить пароль и удалить старые данные",
            "Каков password reset timeout?",
        ],
        "subscription": [
            "Можно приостановить подписку?",
            "На какой срок замораживается тариф?",
            "Ограничения паузы подписки",
            "Приостановить подписку и обратиться в поддержку",
            "Как работает subscription pause?",
        ],
        "invoice": [
            "Где найти счет?",
            "Когда формируется счет?",
            "Доступность счета после оплаты",
            "Я оплатил картой, где счет?",
            "Где отображается invoice?",
        ],
        "support": [
            "Какое время работы приоритетной поддержки?",
            "Когда можно написать в поддержку?",
            "Расписание приоритетной помощи",
            "Часы поддержки и пауза подписки",
            "Когда работает support?",
        ],
        "delete-data": [
            "Как быстро удаляются данные аккаунта?",
            "Когда сотрут мою информацию?",
            "Срок обработки удаления аккаунта",
            "Удалить данные после сброса пароля",
            "Сколько занимает data deletion?",
        ],
    },
    "uz": {
        "cancel-order": [
            "Buyurtmani qachon bekor qilish mumkin?",
            "Xaridni bekor qilsam bo'ladimi?",
            "Buyurtmani bekor qilish shartlari",
            "Buyurtmani bekor qilib yetkazishni ham so'rasam bo'ladimi?",
            "Order cancellation qanday ishlaydi?",
        ],
        "returns": [
            "Qaytarish muddati qancha?",
            "Mahsulotni qaytarish uchun qancha vaqt bor?",
            "Ishlatilmagan mahsulotni qaytarish qoidalari",
            "Mahsulotni qaytarib boshqa buyurtmani bekor qilsam bo'ladimi?",
            "Return period qancha?",
        ],
        "delivery": [
            "Standart yetkazib berish qancha davom etadi?",
            "Posilka qachon keladi?",
            "Oddiy yetkazish muddati",
            "Yetkazib berish va to'lov haqida ayting",
            "Delivery necha kun?",
        ],
        "payment": [
            "Qaysi to'lov usullari qabul qilinadi?",
            "Naqd pul bilan to'lash mumkinmi?",
            "Buyurtma uchun qanday to'lash kerak?",
            "Karta bilan to'lab hisob-faktura olsam bo'ladimi?",
            "Cash payment bormi?",
        ],
        "password": [
            "Tiklash havolasi qancha amal qiladi?",
            "Parol havolasi qachon tugaydi?",
            "Parolni tiklash muddati",
            "Parolni tiklab eski ma'lumotni o'chirish",
            "Password reset timeout qancha?",
        ],
        "subscription": [
            "Obunani to'xtatib turish mumkinmi?",
            "Tarifni qancha vaqtga muzlatish mumkin?",
            "Obuna pauzasi cheklovi",
            "Obunani to'xtatib yordamga murojaat qilish",
            "Subscription pause qanday?",
        ],
        "invoice": [
            "Hisob-fakturani qayerdan topaman?",
            "Hisob qachon yaratiladi?",
            "To'lovdan keyingi faktura",
            "Karta bilan to'ladim, faktura qayerda?",
            "Invoice qayerda ko'rinadi?",
        ],
        "support": [
            "Ustuvor yordam qachon ishlaydi?",
            "Yordamga qachon yozish mumkin?",
            "Yordam xizmati jadvali",
            "Yordam va obuna pauzasi haqida",
            "Support working hours qanday?",
        ],
        "delete-data": [
            "Akkaunt ma'lumoti qachon o'chadi?",
            "Ma'lumotlarim qachon yo'q qilinadi?",
            "Akkauntni o'chirish muddati",
            "Parolni tiklab ma'lumotni o'chirish",
            "Data deletion qancha vaqt?",
        ],
    },
}

UNANSWERABLE = {
    "en": [
        "Do you accept cryptocurrency?",
        "Is there a student discount?",
        "Can drones deliver orders?",
        "Do you ship to Mars?",
        "Can I pay with reward points?",
    ],
    "ru": [
        "Принимаете ли вы криптовалюту?",
        "Есть ли скидка студентам?",
        "Доставляют ли заказы дроны?",
        "Есть ли доставка на Марс?",
        "Можно платить бонусными баллами?",
    ],
    "uz": [
        "Kriptovalyuta qabul qilinadimi?",
        "Talabalar uchun chegirma bormi?",
        "Dron bilan yetkazish bormi?",
        "Marsga yetkazib berasizmi?",
        "Bonus ballari bilan to'lash mumkinmi?",
    ],
}

TAG_BY_VARIANT = ["exact", "paraphrase", "morphology", "multi-intent", "cross-language"]


def main() -> None:
    DOCUMENTS.mkdir(parents=True, exist_ok=True)
    rows = []
    for language, topics in TOPICS.items():
        for topic, title, fact in topics:
            stem = f"{topic}-{language}"
            (DOCUMENTS / f"{stem}.md").write_text(
                f"# {title}\n\n{fact}\n\nThis policy is authoritative for the demo knowledge base.",
                encoding="utf-8",
            )
            for variant, query in enumerate(QUERIES[language][topic]):
                rows.append(
                    {
                        "id": f"{language}-{topic}-{variant + 1:02d}",
                        "group_id": f"{language}:{topic}",
                        "query": query,
                        "language": language,
                        "relevant_chunk_ids": [f"{stem}:0"],
                        "expected_facts": [fact],
                        "answerable": True,
                        "tags": [TAG_BY_VARIANT[variant], topic],
                    }
                )
        for variant, query in enumerate(UNANSWERABLE[language]):
            rows.append(
                {
                    "id": f"{language}-unanswerable-{variant + 1:02d}",
                    "group_id": f"{language}:unanswerable:{variant + 1:02d}",
                    "query": query,
                    "language": language,
                    "relevant_chunk_ids": [],
                    "expected_facts": [],
                    "answerable": False,
                    "tags": ["unanswerable"],
                }
            )
    DATASET.parent.mkdir(parents=True, exist_ok=True)
    DATASET.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} benchmark queries and {len(list(DOCUMENTS.glob('*.md')))} documents")


if __name__ == "__main__":
    main()
