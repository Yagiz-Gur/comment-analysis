import csv
from youtube_comment_downloader import YoutubeCommentDownloader
from src.config import COMMENTS


def save_comments_to_csv(url,status=None):
    downloader = YoutubeCommentDownloader()
    
    # sort by Top comments 
    comments = downloader.get_comments_from_url(url, sort_by=1)

    yield f"Downloading comments from {url}..."

    with open(COMMENTS, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write the header row
        writer.writerow([ 'Text', 'Likes'])

        count = 0
        try:
            for comment in comments:
                writer.writerow([

                    comment['text'],
                    comment['votes']
                ])
                count += 1
                if count % 100 == 0:
                    yield (f"Collected {count} comments so far...")
                    
        except KeyboardInterrupt:
            yield ("\nProcess stopped by user.")
        except Exception as e:
            yield (f"An error occurred: {e}")

    yield (f"\nDone! Saved {count} comments to '{COMMENTS}'.")

