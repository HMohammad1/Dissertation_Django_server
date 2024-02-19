from rest_framework import serializers
from myapp.models import Dissertation

# test commit
class DissertationSerializer(serializers.ModelSerializer):
    # description = serializers.SerializerMethodField()
    class Meta:
        model = Dissertation
        fields = "__all__"
