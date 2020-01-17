using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Threading.Tasks;
using System.Web.Http;
using System.Web.Http.Description;
using WebApplication1.Models;

namespace WebApplication1.Controllers
{
    public class CarsController : ApiController
    {

        // GET: api/Cars
        [ResponseType(typeof(IEnumerable<Car>))]
        public IEnumerable<Car> Get()
        {
            return Data.Cars;
        }

        // GET: api/Data.Cars/5
        public IHttpActionResult Get(int id)
        {
            var product = Data.Cars.FirstOrDefault((p) => p.id == id);
            if (product == null)
            {
                return NotFound();
            }
            return Ok(product);
        }

        // POST: api/Cars
        public void Post([FromBody]string value)
        {

        }

        // PUT: api/Data.Cars/5
        [HttpPut]
        public async Task<IHttpActionResult> Put([FromBody]Car value)
        {
            
            var product = Data.Cars.FirstOrDefault((p) => p.id == value.id);
            if (product == null)
            {
                return NotFound();
            }
            else
            {
                Car car = value;
                await Task.Run(() => Data.Cars[car.id - 1] = car); 
                return Ok();
            }
        }

        // DELETE: api/Cars/5
        public void Delete(int id)
        {
        }
    }

    public class Data
    {
        public static Car[] Cars = new Car[]
        {
        new Car { id = 1, leftspeed = 0, rightspeed= 0, leftlinesensor= false, rightlinesensor=false,ultrasonicsensor=0},
        new Car { id = 2, leftspeed = 0, rightspeed= 0, leftlinesensor= false, rightlinesensor=false,ultrasonicsensor=0},
        new Car { id = 3, leftspeed = 0, rightspeed= 0, leftlinesensor= false, rightlinesensor=false,ultrasonicsensor=0},
        new Car { id = 4, leftspeed = 0, rightspeed= 0, leftlinesensor= false, rightlinesensor=false,ultrasonicsensor=0},
        new Car { id = 5, leftspeed = 0, rightspeed= 0, leftlinesensor= false, rightlinesensor=false,ultrasonicsensor=0}
        };
    }
}
