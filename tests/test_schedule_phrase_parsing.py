"""The Configurator's schedule phrase has to survive the language it was said in.

When the builder commits an agent it turns `task_draft.schedule` — free text
the model wrote down — into a schedule expression, and the caller drops
anything unreadable with a bare `except ValueError: pass`. So a phrase this
cannot parse produces an agent with no schedule and nothing saying why: the
same silent nothing as an agent that was never scheduled at all.

The parser understood English and two Korean forms. "매일 아침 9시" and
"6시간마다" — how people actually say it — fell through.
"""

import sys
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from routes.agents import _schedule_expression_from_draft as parse


class EnglishTest(unittest.TestCase):
    def test_interval_words(self):
        self.assertEqual(parse("every 6 hours"), {"kind": "interval", "seconds": 21600})
        self.assertEqual(parse("every 30 minutes"), {"kind": "interval", "seconds": 1800})
        self.assertEqual(parse("every 2 days"), {"kind": "interval", "seconds": 172800})

    def test_compact_form(self):
        self.assertEqual(parse("every 6h"), {"kind": "interval", "seconds": 21600})

    def test_daily_at(self):
        self.assertEqual(parse("daily at 09:00"), {"kind": "daily_at", "time": "09:00"})
        self.assertEqual(
            parse("every day at 21:30"), {"kind": "daily_at", "time": "21:30"}
        )


class KoreanIntervalTest(unittest.TestCase):
    def test_prefix_form_still_works(self):
        self.assertEqual(parse("매 6시간"), {"kind": "interval", "seconds": 21600})

    def test_suffix_form(self):
        # The common way to say it, and the way the dashboard prints it back.
        self.assertEqual(parse("6시간마다"), {"kind": "interval", "seconds": 21600})
        self.assertEqual(parse("30분마다"), {"kind": "interval", "seconds": 1800})
        self.assertEqual(parse("2일마다"), {"kind": "interval", "seconds": 172800})

    def test_once_per_form(self):
        self.assertEqual(parse("6시간에 한 번"), {"kind": "interval", "seconds": 21600})
        self.assertEqual(parse("2시간 간격"), {"kind": "interval", "seconds": 7200})

    def test_a_sub_minute_interval_is_floored(self):
        # Anything faster than a minute is a runaway loop, not a schedule.
        self.assertEqual(parse("every 1 m"), {"kind": "interval", "seconds": 60})


class KoreanDailyTest(unittest.TestCase):
    def test_clock_form(self):
        self.assertEqual(parse("매일 09:00"), {"kind": "daily_at", "time": "09:00"})

    def test_spoken_hour(self):
        self.assertEqual(parse("매일 아침 9시"), {"kind": "daily_at", "time": "09:00"})
        self.assertEqual(parse("매일 9시"), {"kind": "daily_at", "time": "09:00"})

    def test_afternoon_shifts_to_24_hour(self):
        self.assertEqual(parse("매일 오후 3시"), {"kind": "daily_at", "time": "15:00"})
        self.assertEqual(parse("매일 저녁 8시"), {"kind": "daily_at", "time": "20:00"})

    def test_minutes_are_kept(self):
        self.assertEqual(
            parse("매일 오후 3시 30분"), {"kind": "daily_at", "time": "15:30"}
        )

    def test_the_two_noons(self):
        # 오후 12시 is midday; 오전 12시 is midnight. Adding 12 to both would
        # put one of them a full half-day out.
        self.assertEqual(parse("매일 오후 12시"), {"kind": "daily_at", "time": "12:00"})
        self.assertEqual(parse("매일 오전 12시"), {"kind": "daily_at", "time": "00:00"})

    def test_an_impossible_clock_is_refused(self):
        self.assertIsNone(parse("매일 오후 25시"))


class NoScheduleTest(unittest.TestCase):
    def test_explicit_none(self):
        for text in ("now", "manual", "none", "no schedule", "", "   "):
            self.assertIsNone(parse(text), text)

    def test_korean_none(self):
        for text in ("수동 실행", "직접 실행할게요", "예약 없음"):
            self.assertIsNone(parse(text), text)

    def test_non_strings(self):
        for value in (None, 42, ["every 6 hours"]):
            self.assertIsNone(parse(value))

    def test_unparseable_text_is_not_guessed_at(self):
        # Better no schedule than a schedule the user did not ask for.
        self.assertIsNone(parse("whenever the build breaks"))


if __name__ == "__main__":
    unittest.main()
