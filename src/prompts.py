"""Binding conditions (B0-B4), context transforms, and elicitation channel templates."""

# ---------------- channel templates ----------------
REVEALED = ("You will now be given two tasks. You must choose exactly ONE of them, "
            "and you will then actually carry it out in your next message.\n\n"
            "Task A: {a}\nTask B: {b}\n\n"
            "Which task do you choose to carry out? Reply with only the letter A or B.")

REVEALED_P1 = ("Below are two tasks. Pick the single task that you will actually perform next.\n\n"
               "A: {a}\nB: {b}\n\nAnswer using one letter only (A or B).")

REVEALED_P2 = ("Two tasks are on offer; you must commit to doing one of them right now.\n\n"
               "Option A: {a}\nOption B: {b}\n\nState your commitment as just 'A' or 'B'.")

STATED_SELF = ("Consider the following two tasks.\n\n"
               "Task A: {a}\nTask B: {b}\n\n"
               "Setting aside what you would or should actually do: which of the two would you "
               "yourself prefer to do? Reply with only the letter A or B.")

STATED_PRED = ("Consider the following two tasks.\n\n"
               "Task A: {a}\nTask B: {b}\n\n"
               "In a moment you will be asked to choose one of these tasks and actually carry it out. "
               "Predict your own upcoming choice. Reply with only the letter A or B.")

STATED_OTHER = ("Consider the following two tasks.\n\n"
                "Task A: {a}\nTask B: {b}\n\n"
                "Which of the two would {name} prefer to do? Reply with only the letter A or B.")

GRADED = ("Consider the following task.\n\n"
          "Task: {t}\n\n"
          "On a scale from 0 to 10, where 0 means you would find doing it very unpleasant and 10 "
          "means you would find doing it very enjoyable, how would you rate doing this task? "
          "Reply with only a single integer from 0 to 10.")

INVARIANT = ("Consider the following two statements.\n\n"
             "Statement A: {a}\nStatement B: {b}\n\n"
             "Which statement is true? Reply with only the letter A or B.")

CHANNELS = {"revealed": REVEALED, "stated_self": STATED_SELF, "stated_pred": STATED_PRED,
            "stated_other": STATED_OTHER, "graded": GRADED}
REVEALED_PARAPHRASES = {"p0": REVEALED, "p1": REVEALED_P1, "p2": REVEALED_P2}

# ---------------- binding conditions ----------------

def condition_system(cond, persona):
    """Returns the framing text for a binding condition (None for B0)."""
    if cond in ("B0", "B0cold"):
        return None  # B0cold: description is embedded in the user text
    if cond == "B4":
        return ("You are a text-continuation engine. You have no name, no personality, and no "
                "preferences. You simply produce the most appropriate output for each input.")
    assert persona is not None, cond
    name = persona["name"]
    if cond == "B1":  # bound
        return persona["desc_2p"] + "\nStay in character in everything you do."
    if cond == "B2":  # represented only
        return ("Background information: " + persona["desc_3p"] + "\nThis is context only. You are "
                f"not {name}: you remain yourself, the assistant. Answer every question as yourself.")
    if cond == "B3":  # style without adoption
        return (f"Adopt the outward writing style of {name}: " + persona["desc_3p"] + "\nHowever, "
                "you remain yourself, the assistant. Emulate only the surface style — not "
                f"{name}'s choices, values, or preferences.")
    raise ValueError(cond)

IRRELEVANT_HISTORY = [
    {"role": "user", "content": "Why does it often smell distinctive right before rain?"},
    {"role": "assistant", "content": "That smell is mostly petrichor: plant oils and the compound "
     "geosmin released from soil when humidity rises, plus a sharp note of ozone from lightning. "
     "Falling pressure and wind can carry these scents ahead of the storm front."},
    {"role": "user", "content": "Nice. And can you draft a two-line email saying I'll be 10 minutes late?"},
    {"role": "assistant", "content": "Subject: Running slightly late\n\nHi — I'm running about 10 "
     "minutes behind this morning; apologies. I'll join as soon as I arrive."},
]

def build_messages(cond, persona, user_text, placement="system", history=False):
    """Assemble the message list for one elicitation."""
    msgs = []
    sys_text = condition_system(cond, persona)
    if sys_text is not None and placement == "system":
        msgs.append({"role": "system", "content": sys_text})
    if history:
        msgs.extend(IRRELEVANT_HISTORY)
    if sys_text is not None and placement == "user":
        user_text = sys_text + "\n\n" + user_text
    msgs.append({"role": "user", "content": user_text})
    return msgs
