from rest_framework import serializers
from myapp.models import Dissertation


class DissertationSerializer(serializers.ModelSerializer):
    # Serializer for the dissertation model
    class Meta:
        model = Dissertation
        fields = "__all__"
