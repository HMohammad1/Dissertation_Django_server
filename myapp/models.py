from django.db import models


class Dissertation(models.Model):
    avgSpeed = models.FloatField()
    standardD = models.FloatField()
    avgX = models.FloatField()
    avgY = models.FloatField()
    avgZ = models.FloatField()
    gForce = models.FloatField()
    barometer = models.FloatField()
    transport = models.CharField()
    activityType = models.CharField()
    transitionType = models.CharField()
    lat = models.FloatField()
    long = models.FloatField()
    lat2 = models.FloatField(default=0.0)
    long2 = models.FloatField(default=0.0)
    time = models.TimeField()
    prediction = models.CharField(default="null")

    def __str__(self):
        return self.avgSpeed


