def _ui_add_output_data(file_path,changes,changes_summary,average_summary,changes_apart,item,item_least):
    """
    Writes summarized userbase change data (day/week/month/year) to a given file.

    Args:
        file_path (str): Path to write the output.
        changes (int): Number of periods to review.
        changes_summary (str): Summary of changes in that period.
        average_summary (str): Average of those changes.
        changes_apart (int): Minimum period between change entries.
        item (str): Label for change type.
        item_least (str): Time unit (days/weeks/months/years).
    """
    with open(file_path, "w", encoding="utf-8", errors="ignore") as file:
        file.write(f" ###########{item} Changes###########:\n")
        file.write(f" Changes to your userbase over the past {changes} {item_least}:\n")
        file.write("  " + f"{changes_summary}\n" if changes_summary else "  " + f"None_Found\n")
        file.write(f" Average Changes to your userbase over the past {changes} {item_least}:\n")
        file.write("  " + f"{average_summary}\n" if average_summary else "  " + f"None_Found\n")
        file.write(f" *Only includes userbase changes at least {changes_apart} {item_least} apart.\n")