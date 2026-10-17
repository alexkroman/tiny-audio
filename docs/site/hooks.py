"""MkDocs hooks for the course site."""

from typing import Any

# The course starts at 0-course-overview.md rather than index.md, so the
# README's links keep working; serve that page at the site root.
OVERVIEW = "0-course-overview.md"


def on_files(files: Any, config: Any) -> Any:
    """Publish the overview page as the site's index.html."""
    overview = files.get_file_from_path(OVERVIEW)
    if overview is not None:
        overview.dest_uri = "index.html"
        overview.abs_dest_path = f"{config['site_dir']}/index.html"
        overview.url = "./"
    return files
