"""Short general-purpose passages for the regression check.

Deliberately unrelated to the A -> B -> C tasks. Replace with your own corpus via
`python -m lth.evaluate --general file.jsonl` (one {"text": ...} per line); a few
hundred held-out passages give a much tighter estimate than this built-in set.
"""

GENERAL_TEXTS = [
    "The water cycle moves water between the oceans, the atmosphere and the land. Heat from the sun "
    "evaporates water from the surface, the vapour cools and condenses into clouds, and it returns "
    "to the ground as rain or snow.",
    "To make a simple vinaigrette, whisk one part vinegar with three parts olive oil, then add a "
    "pinch of salt, a little mustard and freshly ground pepper. Taste it and adjust before dressing "
    "the salad.",
    "Photosynthesis is the process by which green plants use sunlight to turn carbon dioxide and "
    "water into glucose and oxygen. It takes place mainly in the leaves, inside structures called "
    "chloroplasts.",
    "The printing press, introduced in Europe in the fifteenth century, made books far cheaper to "
    "produce. Within a few decades millions of volumes were in circulation and literacy began to "
    "spread beyond the clergy and the nobility.",
    "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n\n"
    "print([fibonacci(i) for i in range(10)])",
    "Dear Sam, thank you for coming to the meeting on Tuesday. I have attached the notes and the "
    "revised schedule. Please let me know by Friday whether the new dates work for your team.",
    "A triangle has three sides and three angles, and the angles always add up to one hundred and "
    "eighty degrees. If two of the angles are sixty degrees each, the third must also be sixty "
    "degrees, and the triangle is equilateral.",
    "The old lighthouse stood at the end of the headland, its paint peeling in the salt wind. Every "
    "evening the keeper climbed the spiral stairs, lit the lamp, and watched the fishing boats come "
    "home across the darkening water.",
    "Regular exercise strengthens the heart, improves mood and helps people sleep. Most guidelines "
    "recommend at least one hundred and fifty minutes of moderate activity each week, such as brisk "
    "walking or cycling.",
    "Q: What is the capital of France?\nA: The capital of France is Paris.\n"
    "Q: What is the largest planet in the solar system?\nA: The largest planet is Jupiter.",
    "Supply and demand describe how prices form in a market. When more people want a product than "
    "there are units available, the price tends to rise; when supply exceeds demand, the price "
    "tends to fall.",
    "SELECT name, email FROM customers WHERE created_at >= '2024-01-01' ORDER BY name ASC LIMIT 20;",
    "The human heart has four chambers. The right side pumps blood to the lungs to pick up oxygen, "
    "and the left side pumps the oxygen-rich blood to the rest of the body.",
    "In chess, each player begins with sixteen pieces: one king, one queen, two rooks, two bishops, "
    "two knights and eight pawns. The aim of the game is to checkmate the opponent's king.",
    "Mount Everest, on the border between Nepal and China, is the highest mountain above sea level. "
    "It was first climbed in 1953 by Edmund Hillary and Tenzing Norgay.",
    "To change a flat tyre, first park on level ground and apply the handbrake. Loosen the wheel "
    "nuts slightly, raise the car with the jack, remove the wheel, fit the spare and tighten the "
    "nuts in a star pattern.",
    "Translate to French: Good morning, how are you today?\nBonjour, comment allez-vous aujourd'hui ?",
    "The moon orbits the Earth roughly once every twenty-seven days. Because it rotates at the same "
    "rate, the same face of the moon always points toward us.",
    "Once upon a time there was a fox who lived at the edge of a great forest. Every morning she "
    "walked down to the river to drink, and every morning the heron on the far bank nodded to her.",
    "An index in a database works much like the index of a book. Instead of scanning every row to "
    "find a value, the database looks it up in a sorted structure and jumps straight to the "
    "matching rows.",
    "Bread dough rises because yeast feeds on sugars in the flour and releases carbon dioxide. The "
    "gas is trapped by the stretchy gluten network, and the loaf expands until it is baked.",
    "The meeting is scheduled for Monday at ten o'clock in the main conference room. Please bring a "
    "printed copy of the report and be prepared to summarise your section in five minutes.",
    "Sound travels faster through water than through air, and faster still through solids such as "
    "steel. In air at room temperature its speed is about three hundred and forty metres per second.",
    "Recycling one aluminium can saves most of the energy needed to make a new one from raw ore. "
    "The metal can be melted down and reused again and again without losing its quality.",
]
