
from collections import Counter

def get_call_file_path():
    from extra_models.Sulfur.TrainingScript.Build import call_file_path
    return call_file_path.Call()

call = get_call_file_path()

class InferTimeModel:
    import pandas as pd
    from datetime import datetime, timedelta
    def extract_training_data_with_dates(self, file_path):
        try:
            df = self.pd.read_csv(file_path)
            df.columns = df.columns.str.strip()
            return [
                (
                    row.get('date') if self.pd.notna(row.get('date')) else None,
                    row.get('sentence_type', ''),
                    row.get('intent', ''),
                    row.get('sentence', '')
                )
                for _, row in df.iterrows()
            ]
        except Exception as e:
            print(f"Error extracting training data: {e}")
            return []

    def count_sentences_with_details(self, data, granularity='day'):
        def parse_date(s):
            if not s:  return None
            for fmt in ("%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M", "%Y-%m-%d", "%d/%m/%Y"):
                try:     return self.datetime.strptime(s, fmt).date()
                except Exception:   continue
            raise ValueError(f"Unknown date format: {s}")

        grouped = {}
        for date_str, stype, intent, text in data:
            date_obj = parse_date(date_str)
            if date_obj is None:   continue
            if granularity == 'week':  date_obj -= self.timedelta(days=date_obj.weekday())
            elif granularity == 'month':    date_obj = date_obj.replace(day=1)
            elif granularity == 'year':   date_obj = date_obj.replace(month=1, day=1)
            grouped.setdefault(date_obj, []).append({
                'sentence_type': stype,
                'intent': intent,
                'sentence': text
            })

        return grouped

    def calculate_average_type_and_intent(self, grouped):
        averages = {}
        for date, items in grouped.items():
            total = len(items)
            type_counts = Counter(item['sentence_type'] for item in items)
            intent_counts = Counter(item['intent'] for item in items)

            averages[date] = {
                'average_types': {k: v / total for k, v in type_counts.items()},
                'average_intents': {k: v / total for k, v in intent_counts.items()}
            }

        return averages

    def find_date_pairs(self, averages, min_gap):
        keys = sorted(averages.keys())
        return [
            (keys[i], keys[i + 1])
            for i in range(len(keys) - 1)
            if (keys[i + 1] - keys[i]).days >= int(min_gap)
        ]

    def calculate_change_in_types_and_intents(self, averages, pairs):
        changes = []
        for d1, d2 in pairs:
            t1 = averages[d1]['average_types']
            t2 = averages[d2]['average_types']
            i1 = averages[d1]['average_intents']
            i2 = averages[d2]['average_intents']

            type_changes = {k: t2.get(k, 0) - t1.get(k, 0) for k in t1}
            intent_changes = {k: i2.get(k, 0) - i1.get(k, 0) for k in i1}

            changes.append({
                'period1': d1,
                'period2': d2,
                'type_changes': type_changes,
                'intent_changes': intent_changes
            })

        return changes

    def summarize_changes(self, changes):
        lines = []
        for c in changes:
            type_line = ', '.join(f"{k}: {'+' if v > 0 else ''}{v * 100:.2f}%" for k, v in c['type_changes'].items())
            intent_line = ', '.join(f"{k}: {'+' if v > 0 else ''}{v * 100:.2f}%" for k, v in c['intent_changes'].items())

            lines.append(
                f"Changes from {c['period1']} to {c['period2']}:\n  Type Changes: {type_line}\n  Intent Changes: {intent_line}"
            )

        return '\n'.join(lines)

    def find_average(self, changes):
        type_totals = Counter()
        intent_totals = Counter()
        count = len(changes)

        for change in changes:
            type_totals.update(change['type_changes'])
            intent_totals.update(change['intent_changes'])

        avg_type_change = {k: v / count for k, v in type_totals.items()}
        avg_intent_change = {k: v / count for k, v in intent_totals.items()}

        return {
            'average_type_change': avg_type_change,
            'average_intent_change': avg_intent_change
        }


def run_model(past_d_changes, changes_d_apart_at_least_days, granularity='day'):
    from datetime import datetime, timedelta
    model = InferTimeModel()
    data = model.extract_training_data_with_dates(call.versionDATA_trainingdata_sentences())
    grouped = model.count_sentences_with_details(data, granularity)
    averages = model.calculate_average_type_and_intent(grouped)
    pairs = model.find_date_pairs(averages, past_d_changes)
    changes = model.calculate_change_in_types_and_intents(averages, pairs)
    cutoff = datetime.now().date() - timedelta(days=int(changes_d_apart_at_least_days))
    recent_changes = [c for c in changes if c['period2'] >= cutoff]
    if not recent_changes and changes:   recent_changes = [changes[-1]]
    avg_change = model.find_average(changes)
    average_summary = (
        "\n Average Type Change:\n" +
        '\n'.join(f"  {k}: {'+' if v > 0 else ''}{v * 100:.2f}%" for k, v in avg_change['average_type_change'].items()) +
        "\n\n Average Intent Change:\n" +
        '\n'.join(f"  {k}: {'+' if v > 0 else ''}{v * 100:.2f}%" for k, v in avg_change['average_intent_change'].items())
    )
    changes_summarized = model.summarize_changes(recent_changes)
    average_summary_fixed = "\n\n" + average_summary if changes_summarized else None

    return changes_summarized, average_summary_fixed
