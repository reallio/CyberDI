using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;

namespace demo
{
    internal class Program
    {
        static async void test()
        {
            // Access Token (found from the Account Info Page)
            HttpClient client = new HttpClient();
            client.DefaultRequestHeaders.Add("Authorization", "Token 8f652befe3f7f44bfd8888fbb8be4087923aa59c");

            // Setup Model
            JObject request = new JObject();
            request.Add("project_id", "11");
            HttpContent content = new StringContent(request.ToString());
            content.Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue("application/json");

            var respone = await client.PostAsync("http://localhost:9090/init", content);
            String str = await respone.Content.ReadAsStringAsync();

            while (true)
            {
                DateTime beforeDT = System.DateTime.Now;
                request = new JObject();
                request.Add("project_id", "11");
                request.Add("image", "D:/prj/CybernetiCause/git/CyberDI/workspace/media/upload/11/3952fc6c-NG-U3900036.png");
                request.Add("result_prefix", "D:/prj/CybernetiCause/git/CyberDI/model/clientdemo/testimage/result");
                content = new StringContent(request.ToString());
                content.Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue("application/json");

                respone = await client.PostAsync("http://localhost:9090/eval", content);
                str = await respone.Content.ReadAsStringAsync();
                DateTime afterDT = System.DateTime.Now;
                TimeSpan ts = afterDT.Subtract(beforeDT);
                Console.WriteLine(str);
                Console.WriteLine("DateTime costed: {0}ms", ts.TotalMilliseconds);
            }
            
        }

        static void Main(string[] args)
        {
            test();
            Console.ReadKey(true);
        }
    }
}
