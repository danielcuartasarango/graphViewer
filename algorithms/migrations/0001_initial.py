# Generated by Django 4.0.4 on 2022-04-24 02:21

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Coordenates',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('x', models.IntegerField()),
                ('y', models.IntegerField()),
            ],
        ),
        migrations.CreateModel(
            name='Data',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('ide', models.IntegerField()),
                ('label', models.CharField(max_length=100)),
                ('type', models.CharField(max_length=100)),
                ('radius', models.FloatField()),
                ('coordenates', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='algorithms.coordenates')),
            ],
        ),
        migrations.CreateModel(
            name='Graph',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100)),
                ('data', models.ManyToManyField(to='algorithms.data')),
            ],
        ),
        migrations.CreateModel(
            name='LinkedTo',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('nodeId', models.IntegerField()),
                ('distance', models.IntegerField()),
            ],
        ),
        migrations.CreateModel(
            name='Root',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('generalData1', models.IntegerField()),
                ('generalData2', models.CharField(max_length=100)),
                ('generalData3', models.IntegerField()),
                ('graph', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, to='algorithms.graph')),
            ],
        ),
        migrations.AddField(
            model_name='data',
            name='linkedTo',
            field=models.ManyToManyField(to='algorithms.linkedto'),
        ),
    ]
