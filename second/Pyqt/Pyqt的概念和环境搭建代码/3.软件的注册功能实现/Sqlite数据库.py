"""
    Sqlite数据库

"""
from PyQt5.QtSql import QSqlQuery,QSqlDatabase

class MySql:
    def __init__(self):
        '''
        连接数据库
        '''
        try:
            self.database = QSqlDatabase.addDatabase("QSQLITE")
            self.database.setDatabaseName("user.db")
            self.database.open()
        except Exception as e:
            print(f"无法连接,error:{e}")
    def operation_sql(self,sql):
        self.query = QSqlQuery()
        self.query.exec_(sql)



    def selectData(self,sql):
        '''
        查询
        :return:
        '''
        self.query = QSqlQuery()
        self.query.exec_(sql)
        self.result = self.query.record()
        while self.query.next():
            for i in range(self.result.count()):
                print(str(self.query.value(i)))
    def close_db(self):
        self.db = QSqlDatabase.database()
        self.db.close()
if __name__ == '__main__':
    my_sql = MySql()
    my_sql.operation_sql("create table if not exists test_info(ID int primary key, username text, password text)")
    my_sql.operation_sql("create table if not exists user_info(ID integer primary key AUTOINCREMENT, username text, password text) ")



    #增加
    my_sql.operation_sql("insert into test_info values(1,'test1111111','123456')")
    my_sql.operation_sql("insert into test_info values(2,'test2222222','123456')")
    my_sql.operation_sql("insert into test_info values(null,'test3333333','123456')")

    my_sql.operation_sql("insert into user_info values(null,'test444','123456')")
    my_sql.operation_sql("insert into user_info values(null,'test555','123456')")
    my_sql.operation_sql("insert into user_info values(null,'test666','123456')")
    #
    # 删除
    my_sql.operation_sql("delete from test_info")
    #指定行数删除
    my_sql.operation_sql("delete from user_info where id = 3")



    #更新
    my_sql.operation_sql("update user_info set username = 123456789")
    my_sql.operation_sql("update user_info set username = 111111111 where id = 1")





    #查询
    my_sql.selectData("select * from user_info")


    #关闭
    my_sql.close_db()

