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


class BusesForEachStop(models.Model):
    my_field = models.CharField(db_column='502533031')
    my_field2 = models.CharField(db_column='502532562', default='2, 12, 14, 30, N30')
    my_field3 = models.CharField(db_column='502533030', default='2, 12, 14, 30, N30')
    my_field4 = models.CharField(db_column='502532563', default='2, 12, 14, 30, N30')

    my_field5 = models.CharField(db_column='502532936', default='2, 14, 30, N30')
    my_field6 = models.CharField(db_column='502532670', default='2, 14, 30, N30')
    my_field7 = models.CharField(db_column='502533032', default='2, 14, 30, N30, 12')
    my_field8 = models.CharField(db_column='502532560', default='2, 14, 30, N30, 12')
    my_field9 = models.CharField(db_column='502532559', default='2, 14, 30, N30, 12')
    my_field10 = models.CharField(db_column='502533034', default='2, 14, 30, N30')
    my_field11 = models.CharField(db_column='502532558', default='2, 14, 30, N30')
    my_field12 = models.CharField(db_column='502533035', default='2, 14, 30, N30')
    my_field13 = models.CharField(db_column='502532557', default='2, 14, 30, N30')
    my_field14 = models.CharField(db_column='502533036', default='2, 14, 30, N30')
    my_field15 = models.CharField(db_column='502532556', default='2, 14, 30, N30')
    my_field16 = models.CharField(db_column='502533042', default='2, 14, 30, N30, 33')
    my_field17 = models.CharField(db_column='2700413565', default='2, 14, 30, N30, 33')
    my_field18 = models.CharField(db_column='502533043', default='2, 14, 30, N30, 33')
    my_field19 = models.CharField(db_column='502533044', default='2, 14, 30, N30, 33')
    my_field20 = models.CharField(db_column='502532547', default='2, 14, 30, N30, 33')
    my_field21 = models.CharField(db_column='502533046', default='2, 14, 30, N30, 33')
    my_field22 = models.CharField(db_column='502532542', default='2, 14, 30, N30, 33')
    my_field23 = models.CharField(db_column='502533141', default='14')
    my_field24 = models.CharField(db_column='502532431', default='14')
    my_field25 = models.CharField(db_column='502532428', default='14')
    my_field26 = models.CharField(db_column='502533143', default='14')
    my_field27 = models.CharField(db_column='502533145', default='14')
    my_field28 = models.CharField(db_column='502532425', default='14')
    my_field29 = models.CharField(db_column='502533147', default='14')
    my_field30 = models.CharField(db_column='502532419', default='14')
    my_field31 = models.CharField(db_column='502533150', default='14')
    my_field32 = models.CharField(db_column='502533125', default='2, 30, 33')
    my_field33 = models.CharField(db_column='502532450', default='2, 30, 33')
    my_field34 = models.CharField(db_column='502533126', default='2, 30, 33, 7, 29, 5, 8, 3, 37, 49, 31')
    my_field35 = models.CharField(db_column='502532447', default='47, 37, 7, 31')
    my_field36 = models.CharField(db_column='502532438', default='3, 5, 29, 30, 2, 49, 8, 33')
    my_field37 = models.CharField(db_column='502532436', default='7, 31, 37, 47')
    my_field38 = models.CharField(db_column='502532433', default='5, 29, 30, 49, 8, 2, 3, 33')
    my_field39 = models.CharField(db_column='502533138', default='5, 29, 30, 49, 8, 2, 3, 33, 7, 31, 37')
    my_field40 = models.CharField(db_column='502532417', default='X31, 31, 37, 47, 7')
    my_field41 = models.CharField(db_column='502532413', default='8, 49, 3, 2, X29, 33, 30, 14, 5, 29')
    my_field42 = models.CharField(db_column='502533155', default='31, 30, 29, 5, 33, 14, 37, 3, 8, 49, 7')
    my_field43 = models.CharField(db_column='502533477', default='2')
    my_field44 = models.CharField(db_column='502532411', default='37, 7, 31')
    my_field45 = models.CharField(db_column='502532408', default='33, 30, 8, 3, 45, 29, 5, 49, 35, 14')
    my_field46 = models.CharField(db_column='502533158', default='35, 14, 5, 8, 49, 7, 45')
    my_field47 = models.CharField(db_column='502533161', default='29, 33, 30, 3, 37, 31')

