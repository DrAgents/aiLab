import efweb.web
import asyncio
from init_model import model1, model2
import sqlite3
import config

conn = sqlite3.connect(config.DATABASE)
cursor = conn.cursor()

class Qa(efweb.web.RequestHandler):
    async def post(self):
        query = await self.get_argument('query')
        contain_query = await self.get_argument('contain_query', False)
        num = await self.get_argument('num', 3)

        if contain_query == '1':
            contain_query = True
        if len(query) > 40:
            prediction = model2.prediction(query, cursor, num=int(num), contain_query=contain_query)
            return self.write_json({"success": True, 'result': prediction})
        prediction1 = model1.prediction(query, cursor, num=int(num), contain_query=contain_query)
        if len(query) < 4 and prediction1[0][1] < 0.94:
            return self.write_json({"success": True, 'result': prediction1})
        prediction2 = model2.prediction(query, cursor, num=int(num), contain_query=contain_query)
        prediction = list(sorted(prediction1 + prediction2, key=lambda x: x[1]))[:int(num)]

        return self.write_json({"success": True, 'result': prediction})


loop = asyncio.get_event_loop()
loop.run_until_complete(efweb.web.init_app(loop, host='0.0.0.0', port=8766, routers=[(r'/', Qa)]))
loop.run_forever()
