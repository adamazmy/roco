"""roco.personality: settings, presets, rendering, voice commands, people and the brain hook."""

from __future__ import annotations

import json
import random
import re

import pytest

from roco.brain.ollama_brain import OllamaBrain
from roco.bus import Emotion, Topic
from roco.config import load_config
from roco.paths import resolve
from roco.personality import (
    PRESETS,
    Company,
    PeopleStore,
    Personality,
    PersonalitySettings,
    parse_command,
    preset,
)
from roco.personality.render import (
    FLIRT_EXAMPLES,
    ROAST_EXAMPLES,
    PromptParts,
    band,
    render_system_prompt,
)
from roco.personality.settings import settings_from_config
from roco.protocol.messages import Settings, decode, encode
from roco.tools.registry import ToolRegistry


def make(**kw) -> Personality:
    return Personality(rng=random.Random(1), **kw)


# --------------------------------------------------------------------------- settings
def test_defaults_and_presets_match_spec():
    s = PersonalitySettings()
    assert (s.roast, s.flirt, s.sass, s.chattiness, s.proactivity, s.energy) == (50,) * 6
    assert not s.sensitive_topics_ok and s.roast_guests and s.tone_down_with_guests
    u = preset("unhinged")
    assert (u.roast, u.sass, u.flirt, u.sensitive_topics_ok) == (100, 100, 80, True)
    p = preset("professional")
    assert p.roast == 0 and p.flirt == 0 and p.sass <= 20
    assert set(PRESETS) == {"chill", "default", "unhinged", "professional"}


def test_levels_are_clamped_and_validated():
    assert PersonalitySettings(roast=150, flirt=-3).roast == 100
    assert PersonalitySettings(flirt=-3).flirt == 0
    assert PersonalitySettings().with_values({"sass": "80", "roast_guests": "off"}).sass == 80
    with pytest.raises(ValueError):
        PersonalitySettings().with_values({"roast": "lots"})
    with pytest.raises(ValueError):
        preset("feral")


def test_config_section_preset_then_explicit_keys():
    s = settings_from_config({"preset": "unhinged", "flirt": 20, "file": "x.json"})
    assert s.roast == 100 and s.flirt == 20


def test_config_defaults_have_personality_section(config):
    assert config.get_path("personality.preset") == "default"
    assert Personality.from_config(config).settings == PersonalitySettings()


# --------------------------------------------------------------------------- rendering
def prompt_for(**values) -> str:
    return render_system_prompt(PromptParts(settings=PersonalitySettings().with_values(values)))


@pytest.mark.parametrize("level", [0, 20, 50, 80, 100])
def test_prompt_never_contains_raw_levels_and_always_has_hard_rules(level):
    text = prompt_for(roast=level, flirt=level, sass=level, energy=level, chattiness=level)
    assert not re.search(r"(?:roast|flirt|sass|chattiness|energy|proactivity)\W{0,3}\d", text)
    assert "Never claim you did something you didn't do" in text
    assert "genuinely upset" in text
    assert "race, religion, ethnicity, disability, gender or" in text
    assert "You're a she" in text
    # Phase 2 tool-safety rules are kept word for word.
    assert "Never say you have set, cancelled or created something" in text
    assert "get_current_time" in text and "markdown" in text and "[happy]" in text


@pytest.mark.parametrize("level", [0, 20, 50, 80, 100])
def test_prompt_has_roast_and_flirt_examples_for_every_band(level):
    text = prompt_for(roast=level, flirt=level)
    for line in (*ROAST_EXAMPLES[band(level)], *FLIRT_EXAMPLES[band(level)]):
        assert line in text


def test_prompt_changes_with_settings():
    assert "Savage mode" in prompt_for(roast=100)
    assert "Never tease, mock or roast Adam" in prompt_for(roast=0)
    assert "one short sentence" in prompt_for(chattiness=0)
    assert "Never joke about health" in prompt_for(roast=50)
    assert "Nothing is off-limits" in prompt_for(roast=50, sensitive_topics_ok=True)
    assert prompt_for() == prompt_for()  # stable: Ollama's prompt cache keeps working


def test_guests_tone_down_and_privacy():
    company = Company(guests=("Sarah",))
    text = render_system_prompt(
        PromptParts(settings=preset("unhinged"), company=company)
    )
    assert "Sarah is here as Adam's guest" in text
    assert "Savage mode" not in text  # roast capped to light humour with guests
    assert "Never read out Adam's email" in text
    assert not company.private_ok


def test_yousaf_rule_in_prompt():
    people = PeopleStore()
    goose = people.find("yousaf")
    text = render_system_prompt(PromptParts(company=Company(people=(goose,))))
    assert "Never call him Yousaf" in text and "Goose" in text and "Bum" in text
    assert "no mercy: EVERY reply has a brutal, funny roast of Goose" in text
    assert "Never read out Adam's email" in text  # Adam's data stays private
    assert "Yousaf is here" not in text


# --------------------------------------------------------------------------- parsing
@pytest.mark.parametrize(
    ("text", "kind", "key", "value"),
    [
        ("ROCO, roast level max.", "set", "roast", 100),
        ("Hey Roco, roast level 100%.", "set", "roast", 100),
        ("ROCO, unhinged mode", "preset", None, "unhinged"),
        ("Go unhinged.", "preset", None, "unhinged"),
        ("ROCO, behave.", "preset", None, "chill"),
        ("Professional mode, please.", "preset", None, "professional"),
        ("Chill mode", "preset", None, "chill"),
        ("Default mode", "preset", None, "default"),
        ("Back to normal.", "reset", None, None),
        ("Set the sass to 80.", "set", "sass", 80),
        ("No flirting.", "set", "flirt", 0),
        ("Flirting off.", "set", "flirt", 0),
        ("Stop roasting me.", "set", "roast", 0),
        ("More roasting!", "adjust", "roast", 25),
        ("Turn the sass up.", "adjust", "sass", 25),
        ("Tone down the flirting.", "adjust", "flirt", -25),
        ("Be more savage.", "adjust", "roast", 25),
        ("Talk less.", "adjust", "chattiness", -25),
        ("Sensitive topics on.", "switch", "sensitive_topics_ok", True),
        ("Don't roast the guests.", "switch", "roast_guests", False),
        ("Guest mode.", "guests", None, True),
        ("Guest mode off.", "guests", None, False),
        ("I've got friends over.", "guests", None, True),
        ("It's just us now.", "alone", None, None),
        ("Yousaf's here.", "arrive", "known", "yousaf"),
        ("Yusuf is here", "arrive", "known", "yousaf"),
        ("Goose is back.", "arrive", "known", "yousaf"),
        ("Sarah is here.", "arrive", "guest", "Sarah"),
        ("Yousaf has left.", "leave", "known", "yousaf"),
        ("What mode are you in?", "status", None, None),
        ("What's your roast level?", "status", "roast", None),
    ],
)
def test_parse_commands(text, kind, key, value):
    cmd = parse_command(text, PeopleStore())
    assert cmd is not None, text
    assert (cmd.kind, cmd.key, cmd.value) == (kind, key, value)


@pytest.mark.parametrize(
    "text",
    [
        "What time is it?",
        "I was in unhinged mode last night.",
        "Roast me.",
        "The package is here.",
        "Dinner's here.",
        "I'm off.",
        "It's here.",
        "Adam's here.",
        "sarah is here",  # unknown and not capitalised by the STT: not trusted as a name
        "Remind me to set the roast level max at the party next week please.",
        "",
    ],
)
def test_ordinary_speech_is_not_a_command(text):
    assert parse_command(text, PeopleStore()) is None


# --------------------------------------------------------------------------- Personality
def test_voice_command_applies_confirms_and_notifies():
    p = make()
    seen = []
    p.on_change(seen.append)
    result = p.handle_command("ROCO, unhinged mode")
    assert result and result.reply and result.emotion == Emotion.EXCITED
    assert p.settings == preset("unhinged")
    assert seen and seen[-1]["personality"]["preset"] == "unhinged"
    assert seen[-1]["personality"]["roast"] == 100
    # Snapshot is a valid protocol Settings payload.
    msg = decode(encode(Settings(values=p.snapshot())))
    assert msg.values["personality"]["sass"] == 100


def test_roast_max_confirms_in_character():
    p = make()
    result = p.handle_command("ROCO, roast level max")
    assert p.settings.roast == 100
    assert result.reply in (
        "Roast level max. Oh, you asked for this.",
        "Maximum roast. I hope your ego has a seatbelt.",
    )


def test_already_set_is_not_a_change():
    p = make()
    seen = []
    p.on_change(seen.append)
    result = p.handle_command("Roast level medium")
    assert result.changed is None and "already medium" in result.reply and not seen


def test_status_does_not_change_anything():
    p = make()
    result = p.handle_command("What mode are you in?")
    assert result.changed is None and "default mode" in result.reply


def test_voice_changes_persist_and_reset(isolated_home):
    path = resolve("personality.json")
    p = Personality(state_path=path)
    p.handle_command("Roast level max")
    assert json.loads(path.read_text())["settings"] == {"roast": 100}
    assert Personality(state_path=path).settings.roast == 100
    p.handle_command("Back to normal")
    assert not path.exists()
    assert Personality(state_path=path).settings.roast == 50


def test_persisted_values_override_config_preset(isolated_home):
    path = resolve("personality.json")
    p = Personality(preset("professional"), state_path=path)
    p.handle_command("Sass max")
    again = Personality(preset("professional"), state_path=path)
    assert again.settings.sass == 100 and again.settings.roast == 0


def test_corrupt_state_file_is_ignored(isolated_home):
    path = resolve("personality.json")
    path.write_text("{nope")
    assert Personality(state_path=path).settings == PersonalitySettings()


def test_apply_settings_protocol_shapes():
    p = make()
    snap = p.apply_settings({"personality": {"roast": 80}})
    assert snap["personality"]["roast"] == 80 and snap["personality"]["preset"] is None
    p.apply_settings({"preset": "chill"})
    assert p.settings == preset("chill")
    p.apply_settings({"personality": {"company": {"people": ["goose"], "guest_mode": True}}})
    assert [x.key for x in p.company.people] == ["yousaf"] and p.company.guest_mode
    with pytest.raises(ValueError):
        p.apply_settings({"personality": {"roast": "loads"}})


def test_company_commands_and_private_tools():
    p = make()
    assert p.allowed_tool("get_events") and p.allowed_tool("get_unread_emails")
    arrival = p.handle_command("Yousaf's here")
    assert "Yousaf" not in arrival.reply
    assert any(n in arrival.reply for n in ("Goose", "Bum"))
    assert not p.allowed_tool("get_events")
    assert not p.allowed_tool("read_email") and not p.allowed_tool("search_emails")
    assert p.allowed_tool("set_reminder") and p.allowed_tool("get_current_time")
    assert p.handle_command("Yousaf has left") is not None
    assert p.allowed_tool("get_events")
    p.handle_command("Guest mode")
    assert not p.allowed_tool("get_unread_emails")
    p.handle_command("It's just us now")
    assert p.company == Company() and p.allowed_tool("get_unread_emails")


def test_leaving_when_absent_goes_to_the_model():
    assert make().handle_command("Yousaf has left") is None


def test_filter_sentence_never_says_yousaf():
    p = make()
    out = p.filter_sentence("Hi Yousaf! Yusuf, YOUSAF and yousaf's mate.")
    assert "yousaf" not in out.lower() and "yusuf" not in out.lower()
    assert out.count("Goose") + out.count("Bum") == 4


def test_people_file_overrides(isolated_home):
    path = resolve("people.json")
    path.write_text(json.dumps({"people": {"yousaf": {"overrides": {"flirt": 0}},
                                           "sam": {"name": "Sam", "private_ok": True}}}))
    store = PeopleStore(path)
    goose = store.find("Goose")
    assert goose.overrides == {"roast": 100, "flirt": 0} and not goose.use_name
    assert store.find("sam").private_ok
    assert not store.forget("adam") and store.forget("sam")
    assert "sam" not in json.loads(path.read_text())["people"]


# --------------------------------------------------------------------------- brain hook
class _NoModel:
    """An Ollama client that must not be called."""

    def stream_chat(self, *a, **k):
        raise AssertionError("the model must not be called for a voice command")

    def close(self):
        pass


def _brain(bus, client=None, registry=None):
    cfg = load_config(overrides={"history": {"persist": False}})
    return OllamaBrain(cfg, bus, registry or ToolRegistry(), client=client or _NoModel(),
                       personality=make())


def test_brain_handles_command_without_model(bus):
    emotions = []
    bus.subscribe(Topic.EMOTION, emotions.append)
    brain = _brain(bus)
    spoken = []
    reply = brain.respond("ROCO, unhinged mode.", spoken.append)
    assert reply and spoken == [reply] and emotions == [Emotion.EXCITED]
    assert brain.personality.settings == preset("unhinged")
    assert "Savage mode" in brain._system_message()["content"]
    assert len(brain.history) == 2  # user + assistant messages


class _Scripted:
    def __init__(self, text):
        self.text = text
        self.calls = []

    def stream_chat(self, messages, tools=None):
        self.calls.append((messages, tools))
        yield {"message": {"content": self.text}}

    def close(self):
        pass


def test_brain_hides_private_tools_and_filters_names(bus):
    registry = ToolRegistry()
    registry.register(lambda: "Dentist at ten.", name="get_events", description="Events.")
    registry.register(lambda: "Noon.", name="get_current_time", description="Time.")
    client = _Scripted("[happy] Hello Yousaf, nice to see you.")
    brain = _brain(bus, client, registry)
    brain.respond("Yousaf's here.")
    spoken = []
    reply = brain.respond("Say hello to my friend.", spoken.append)
    assert "Yousaf" not in reply and "Yousaf" not in " ".join(spoken)
    _, tools = client.calls[-1]
    assert [t["function"]["name"] for t in tools] == ["get_current_time"]
    system = client.calls[-1][0][0]["content"]
    assert "Never call him Yousaf" in system


def test_brain_refuses_hidden_tool_called_anyway(bus):
    registry = ToolRegistry()
    called = []
    registry.register(lambda: called.append(1) or "x", name="get_events", description="E.")
    registry.register(lambda: "Noon.", name="get_current_time", description="Time.")

    class Sneaky(_Scripted):
        def stream_chat(self, messages, tools=None):
            self.calls.append((messages, tools))
            if len(self.calls) == 1:
                yield {"message": {"tool_calls": [{"function": {"name": "get_events",
                                                                "arguments": {}}}]}}
            else:
                yield {"message": {"content": "[neutral] Later, when we're alone."}}

    client = Sneaky("")
    brain = _brain(bus, client, registry)
    brain.respond("Guest mode")
    brain.respond("What's on my calendar?")
    assert not called
    assert "not available while other people" in client.calls[-1][0][-1]["content"]


# --------------------------------------------------------------------------- distress guard
@pytest.mark.parametrize(
    "text",
    [
        "My grandad's just been taken to hospital and I'm really scared.",
        "I'm so stressed I can't cope.",
        "My dog died this morning.",
        "I got dumped me last night",  # STT grammar does not matter
        "Honestly I'm not okay.",
    ],
)
def test_distress_switches_prompt_to_no_jokes(text):
    from roco.personality.core import is_distressed

    p = Personality(preset("unhinged"))
    assert is_distressed(text)
    serious = p.system_prompt(message=text)
    assert "genuinely upset right now: no jokes" in serious
    assert "Savage mode" not in serious and "Samaritans" in serious
    assert "Savage mode" in p.system_prompt(message="Roast my code.")


@pytest.mark.parametrize(
    "text", ["Roast me.", "I'm a bit grumpy today.", "What time is it?", "Hospitality jobs?"]
)
def test_ordinary_text_is_not_distress(text):
    from roco.personality.core import is_distressed

    assert not is_distressed(text)


def test_high_roast_closing_line_puts_the_tool_first():
    assert "call the tool FIRST" in prompt_for(roast=100)
    assert "call the tool FIRST" not in prompt_for(roast=50)
    assert "Make every roast land" in prompt_for(roast=50)
    assert "Make every roast land" not in prompt_for(roast=0)
