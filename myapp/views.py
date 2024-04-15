import json

from django.db import connection
from django.http import JsonResponse
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from myapp.serealizer import DissertationSerializer
import pandas as pd


# Posts a new record into the database
class Create(APIView):
    def post(self, request):
        serializer = DissertationSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# calculates the distance between two geometries
class Distance(APIView):
    def post(self, request):
        import json

        data = json.loads(request.body)
        latitude = data.get('latitude')
        longitude = data.get('longitude')

        query = """
            SELECT ST_Distance(
                ST_MakePoint(%s, %s)::geography,
                '0101000020E61000000C3444BA447A09C0E1C9B8F30EF94B40'::geography
            ) AS distance_meters;
        """

        with connection.cursor() as cursor:
            cursor.execute(query, [longitude, latitude])
            row = cursor.fetchone()

        if row:
            return JsonResponse({"distance_meters": row[0]})
        else:
            return JsonResponse({"error": "Could not calculate distance"}, status=400)


# checks what bus stops are near the user
class NearbyTransport(APIView):
    def post(self, request):
        data = json.loads(request.body)
        latitude = data.get('latitude')
        longitude = data.get('longitude')
        radius = data.get('radius', 15)

        query = ("\n"
                 "            SELECT *\n"
                 "            FROM transport_free_1\n"
                 "            WHERE ST_DWithin(\n"
                 "                geom::geography,\n"
                 "                ST_MakePoint(%s, %s)::geography,\n"
                 "                %s\n"
                 "            );\n"
                 "        ")

        results = []
        with connection.cursor() as cursor:
            cursor.execute(query, [longitude, latitude, radius])
            columns = [col[0] for col in cursor.description]
            for row in cursor.fetchall():
                results.append(dict(zip(columns, row)))

        if results:
            return JsonResponse({"results": results})
        else:
            return JsonResponse({"error": "No nearby transports found"}, status=404)


# predicts the users transport (deprecated)
class PredictTransport(APIView):
    def post(self, request):
        all_data = pd.read_csv('all_data_1.csv').drop(
            columns=['id', 'activityType', 'transitionType', 'lat', 'long', 'time', 'lat2', 'long2', 'prediction'])
        all_data = all_data[~all_data['transport'].isin(['walking', 'still', 'walking ', 'testing', 'test'])]

        randomized_data = all_data.sample(frac=1, random_state=42)

        X = randomized_data.drop(columns=['transport'])
        y = randomized_data['transport']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)

        try:
            data = json.loads(request.body)
            speed = data.get('speed')
            standardD = data.get('standardD')
            avgX = data.get('avgX')
            avgY = data.get('avgY')
            avgZ = data.get('avgZ')
            gForce = data.get('gForce')
            bar = data.get('bar')
            prediction = clf.predict([[speed, standardD, avgX, avgY, avgZ, gForce, bar]])
            response_data = {
                'prediction': prediction[0]
            }
            return JsonResponse(response_data)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)


# checks if the user is on a bus route
class OnBusRoute(APIView):
    def post(self, request):
        data = json.loads(request.body)
        latitude = data.get('latitude')
        longitude = data.get('longitude')
        radius = data.get('radius', 10)

        # SQL query to find nearby bus routes
        query = """
            SELECT name
            FROM public."BusRoutes"
            WHERE ST_DWithin(
                geom::geography,
                ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography,
                %s
            );
        """

        results = []
        with connection.cursor() as cursor:
            cursor.execute(query, [longitude, latitude, radius])
            columns = [col[0] for col in cursor.description]
            for row in cursor.fetchall():
                results.append(dict(zip(columns, row)))

        if results:
            return JsonResponse({"results": results})
        else:
            return JsonResponse({"error": "No nearby bus routes found"}, status=404)


# get the buses that go to a particular bus stop
class FetchBusData(APIView):
    def post(self, request):
        data = json.loads(request.body)
        column_name = data.get('column_name')

        query = f"""
            SELECT string_to_array("{column_name}", ', ') 
            FROM public.myapp_busesforeachstop;
        """

        results = []
        with connection.cursor() as cursor:
            cursor.execute(query)
            for row in cursor.fetchall():
                results.append(row[0])

        if results:
            return JsonResponse({"results": results})
        else:
            return JsonResponse({"error": "No data found"}, status=404)
