from django.db import models

# Create your models here.
class Cart(models.Model):
    use_in_migration = True
    cart_id = models.AutoField(primary_key=True)

    class Meta:
        db_table = 'shop_carts'
    def __str__(self):
        return f'{self.pk}'