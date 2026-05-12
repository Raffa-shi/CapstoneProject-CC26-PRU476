import instaloader
import pandas as pd

# Inisialisasi
L = instaloader.Instaloader()

# OPTIONAL LOGIN
# Sangat disarankan agar tidak cepat diblok
L.login("moodyheady3", "$moodysaga1")

# URL post Instagram
post_shortcode = "DO6EIPujOYw"

# Ambil post
post = instaloader.Post.from_shortcode(L.context, post_shortcode)

comments_data = []

# Loop comments
for comment in post.get_comments():
    comments_data.append({
        "username": comment.owner.username,
        "comment": comment.text,
        "likes": comment.likes_count,
        "created_at": comment.created_at_utc
    })

# Convert ke DataFrame
comments_df = pd.DataFrame(comments_data)

# Save CSV
comments_df.to_csv("instagram_comments.csv", index=False)

print(comments_df.head())
print(f"Total comments: {len(comments_df)}")