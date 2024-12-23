import os
import shutil
import sys


def move_run_to_archive(run_id):
    # Define the root directories
    runs_root = 'runs'
    archive_root = 'runs_archive'

    # Walk through the runs directory to find the specified run_id
    for root, dirs, files in os.walk(runs_root):
        for dir_name in dirs:
            if dir_name == run_id:
                # Construct the full path to the source directory
                source_dir = os.path.join(root, dir_name)

                # Construct the relative path from the runs root
                relative_path = os.path.relpath(source_dir, runs_root)

                # Construct the full path to the destination directory
                dest_dir = os.path.join(archive_root, relative_path)

                # Create the destination directory if it doesn't exist
                os.makedirs(dest_dir, exist_ok=True)

                # Move the contents of the source directory to the destination directory
                for item in os.listdir(source_dir):
                    s = os.path.join(source_dir, item)
                    d = os.path.join(dest_dir, item)
                    if os.path.isdir(s):
                        shutil.move(s, d)
                    else:
                        shutil.move(s, d)

                # Optionally, remove the now-empty source directory
                os.rmdir(source_dir)

                print(f"Moved {source_dir} to {dest_dir}")
                return

    print(f"Run ID {run_id} not found in {runs_root}")


if __name__ == "__main__":

    runs_to_archive = [
        # '2024-06-13--11-52-22',
        # '2024-06-14--18-12-39',
        # '2024-06-15--18-22-56',
        # '2024-06-16--13-48-58',
        # '2024-06-15--23-10-40',
        # '2024-06-16--14-29-48',
        # '2024-06-16--10-03-52',
        # '2024-06-17--10-06-49',
        # '2024-06-17--15-05-45',
        # '2024-06-14--08-43-51',
        # '2024-06-18--19-19-26',
        # '2024-06-18--20-29-21',
        # '2024-06-17--17-32-44',
        # '2024-06-16--15-46-56',
        # '2024-06-19--15-55-33',
        # '2024-06-18--21-41-57',
        # '2024-06-21--15-21-27',

        "2024-07-12--02-52",
        "2024-07-14--23-28",
        "2024-07-14--21-24",
        "2024-07-13--08-53",
    ]

    for run_id in runs_to_archive:
        move_run_to_archive(run_id)