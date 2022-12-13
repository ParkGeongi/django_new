# Generated by Django 4.1.4 on 2022-12-13 09:00

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="Comment",
            fields=[
                ("comment_id", models.AutoField(primary_key=True, serialize=False)),
                ("content", models.TextField()),
                ("created_at", models.DateTimeField()),
                ("updated_at", models.DateTimeField()),
            ],
            options={
                "db_table": "blog_comments",
            },
        ),
    ]
