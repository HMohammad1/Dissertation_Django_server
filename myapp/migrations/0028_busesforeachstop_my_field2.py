# Generated by Django 5.0.1 on 2024-02-23 15:36

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('myapp', '0027_busesforeachstop'),
    ]

    operations = [
        migrations.AddField(
            model_name='busesforeachstop',
            name='my_field2',
            field=models.CharField(db_column='502532562', default='2, 12, 14, 30, N30'),
        ),
    ]
